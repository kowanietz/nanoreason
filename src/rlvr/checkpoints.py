"""RLVR manifests and strict resumption of the latest complete FSDP checkpoint."""

import hashlib
import json
from pathlib import Path

import torch.distributed as dist


def validate_checkpoint(path: Path, world: int) -> dict:
    """Validate counters and referenced DCP shard extents without loading weights."""
    import torch
    from torch.distributed.checkpoint import FileSystemReader

    required = [
        "trainer_state.json",
        "scheduler.pt",
        "optimizer_0/.metadata",
        "pytorch_model_fsdp_0/.metadata",
        *[f"rng_state_{rank}.pth" for rank in range(world)],
    ]
    if (path / "checkpoint-is-incomplete.txt").exists() or any(
        not (path / name).is_file() or not (path / name).stat().st_size
        for name in required
    ):
        raise ValueError(f"Incomplete native FSDP checkpoint: {path}")
    state = json.loads((path / "trainer_state.json").read_text())
    step = state["global_step"]
    scheduler = torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True)
    if step < 1 or path.name != f"checkpoint-{step}" or scheduler["last_epoch"] != step:
        raise ValueError(f"Inconsistent checkpoint step counters: {path}")
    for folder in ("pytorch_model_fsdp_0", "optimizer_0"):
        directory = path / folder
        metadata = FileSystemReader(directory).read_metadata()
        if not metadata.storage_data:
            raise ValueError(f"Empty DCP checkpoint: {directory}")
        sizes = {}
        for item in metadata.storage_data.values():
            shard = directory / item.relative_path
            if shard not in sizes:
                sizes[shard] = shard.stat().st_size if shard.is_file() else -1
            if sizes[shard] < item.offset + item.length:
                raise ValueError(f"Missing or truncated DCP shard: {shard}")
    return state


def fingerprint(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resume_checkpoint(
    output: Path, setting: str | bool | None, world: int
) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    latest = get_last_checkpoint(str(output)) if output.is_dir() else None
    if setting in (None, False):
        if latest:
            raise ValueError("Existing checkpoints require resumption")
        return None
    checkpoint = latest if setting == "auto" else str(Path(setting).resolve())
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    if (
        path.resolve().parent != output.resolve()
        or path.resolve() != Path(latest or "").resolve()
    ):
        raise ValueError(f"Incomplete or non-latest native FSDP checkpoint: {path}")
    validate_checkpoint(path, world)
    return str(path)


def write_run_manifest(
    output: Path, checkpoint: str | None, settings: dict, config: dict, rank: int
) -> None:
    """Validate resume provenance before any checkpoint or output is written."""
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["sha256"] != digest:
            raise ValueError(
                "Training configuration or inputs changed since the first job"
            )
    elif checkpoint or (output.exists() and any(output.iterdir())):
        raise ValueError("Refusing an unmanifested output directory")
    dist.barrier()
    if not manifest_path.exists() and rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {"sha256": digest, "settings": settings, "resolved_config": config},
                indent=2,
            )
        )
    dist.barrier()
