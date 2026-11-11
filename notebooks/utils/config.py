MODELS = {
    "Base": "Olmo-3-7B",
    "SFT": "NanoReason-7B-SFT",
    "Refined": "NanoReason-7B-SFT-Refined",
    "Think (RLVR)": "NanoReason-7B-Think",
}

STAGES = list(MODELS)

BENCHMARKS = ["gpqa", "math500", "aime-2024", "aime-2025", "bbh", "ifeval"]

TITLES = {
    "gpqa": "GPQA Main",
    "math500": "MATH-500",
    "aime-2024": "AIME 2024",
    "aime-2025": "AIME 2025",
    "bbh": "BBH",
    "ifeval": "IFEval",
}

METRIC_LABELS = {
    "gpqa": "Exact match - 448 questions",
    "math500": "Flexible exact match - 500 problems",
    "aime-2024": "Pass@1 - 30 problems × 32 samples",
    "aime-2025": "Pass@1 - 30 problems × 32 samples",
    "bbh": "Exact match, micro - 6,511 examples",
    "ifeval": "Prompt-level loose - 541 prompts",
}

PRIMARY_METRICS = {
    "gpqa": "exact_match",
    "math500": "exact_match_flex",
    "aime-2024": "pass_at_1",
    "aime-2025": "pass_at_1",
    "bbh": "primary_score_micro",
    "ifeval": "prompt_level_loose_acc",
}

# Base -> SFT -> Refined -> Think (RLVR)
# TODO better colors?
COLORS = dict(zip(STAGES, ["#858E98", "#3579AD", "#219585", "#BB7734"]))

PASS_K = [1, 4, 8, 16, 32]
