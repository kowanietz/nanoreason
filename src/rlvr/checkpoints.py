"""RLVR manifests and strict resumption of the latest complete FSDP checkpoint."""

import hashlib
import json
from pathlib import Path

import torch.distributed as dist


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
    required = [
        "trainer_state.json",
        "scheduler.pt",
        "optimizer_0/.metadata",
        "pytorch_model_fsdp_0/.metadata",
        *[f"rng_state_{rank}.pth" for rank in range(world)],
    ]
    if (
        path.resolve().parent != output.resolve()
        or path.resolve() != Path(latest or "").resolve()
        or (path / "checkpoint-is-incomplete.txt").exists()
        or any(not (path / name).is_file() for name in required)
    ):
        raise ValueError(f"Incomplete or non-latest native FSDP checkpoint: {path}")
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
