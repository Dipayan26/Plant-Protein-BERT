# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

Plant-Protein-BERT: domain-specific BERT pretraining on plant protein sequences.
Stack: PyTorch, PyTorch Lightning, HuggingFace Transformers, Hydra, WandB.

## Setup

```bash
pip install -e ".[dev]"
```

## Build / Lint / Test Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_parser.py

# Run a single test by name
pytest -k "test_parse_yields_records"

# Run a single test function directly
pytest tests/test_parser.py::test_parse_yields_records

# Lint (ruff)
ruff check src/ tests/ scripts/

# Lint with auto-fix
ruff check --fix src/ tests/ scripts/

# Type check
mypy src/

# Run a training pipeline step
python scripts/pretrain.py +experiment=dev_run
python scripts/finetune.py +experiment=finetune_go_terms training.pretrained_checkpoint=<path>
python scripts/evaluate.py training.pretrained_checkpoint=<path>
```

## Code Style

### General

- Python 3.10+; use `from __future__ import annotations` at the top of every module.
- Line length: 100 characters (configured in `pyproject.toml`).
- Ruff rules enabled: `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `UP` (pyupgrade).
- `E501` (line too long) is ignored — rely on the 100-char soft limit.
- Type checking with mypy (`warn_return_any = true`, `strict = false`, `ignore_missing_imports = true`).

### Imports

- Use `from __future__ import annotations` as the first code line (after the module docstring).
- Standard library imports first, then third-party, then local — enforced by ruff `I` rule.
- Use relative imports within the `plant_bert` package (e.g., `from ..models.heads import SequenceClassificationHead`).
- Absolute imports in scripts (e.g., `from plant_bert.utils import seed_everything`).

### Types

- Use modern union syntax: `str | Path`, `torch.Tensor | None` (not `Optional` or `Union`).
- Use `list[str]`, `dict[str, Any]` lowercase generics (not `List`, `Dict`).
- Return type annotations on all public functions and methods.
- Use `-> None` on `__init__`, `training_step`, etc.
- Use `object` for opaque Hydra-instantiated dependencies (e.g., `model: object`, `tokenizer: object`).

### Naming

- Classes: `PascalCase` (e.g., `PlantProteinBERT`, `StreamingProteinDataset`, `MLMPretrainer`).
- Functions/methods: `snake_case` (e.g., `parse_dat_gz`, `get_sequence_embedding`).
- Constants: `UPPER_SNAKE_CASE` (e.g., `_OX_RE` for module-private compiled regexes).
- Private: prefix with single underscore (e.g., `_hf`, `_forward`, `_length`).
- Test functions: `test_` prefix, descriptive names (e.g., `test_parse_yields_records`).
- Config keys: `snake_case` in YAML, matching Python parameter names.

### Dataclasses

- Use `@dataclass` from the standard library for structured records (e.g., `UniProtRecord`).
- Use `field(default_factory=list)` for mutable defaults, never `[]` directly.

### Docstrings

- Module-level docstrings required — explain the module's purpose and any non-obvious design decisions.
- Multi-line docstrings on classes and public functions using triple-double-quotes.
- Docstring style: prose paragraphs, not Google/NumPy style parameter blocks (though Args/Yields/Returns sections are used in the parser).
- Inline comments explain *why*, not *what*.

### Error Handling

- Let exceptions propagate naturally — do not wrap in broad `try/except` blocks.
- Use `errors="replace"` when opening files with potential encoding issues (see `uniprot_parser.py`).
- Validate data at parse time, not downstream.

### Architecture Constraints (strict layering)

```
utils ← tokenizer ← data ← models ← training / evaluation
```

No upward imports. `models` never imports from `training`. `data` never imports from `models`. This is enforced by convention — respect it.

### Hydra / Config

- All training scripts are Hydra entry points. Config defaults live in `configs/config.yaml`.
- Use `instantiate()` from `hydra.utils` to build objects from config.
- Use `_recursive_=False` when a config node should be passed as-is rather than instantiated.
- Override any config value via CLI: `python scripts/pretrain.py model=bert_large training.trainer.devices=-1`.
- Every run auto-snapshots config to `outputs/YYYY-MM-DD/HH-MM-SS/.hydra/`.

### Testing

- Tests use synthetic in-memory data — never depend on the 18 GB TrEMBL file.
- Shared fixtures go in `tests/conftest.py`.
- Fixtures use `tmp_path` (pytest built-in) for temporary files.
- Prefer `assert` statements over `unittest`-style assertions.

### PyTorch / Lightning Conventions

- `LightningModule` subclasses for training logic (`MLMPretrainer`, `FineTuner`).
- Use `self.log()` with `sync_dist=True` for multi-GPU metric logging.
- Use `self.save_hyperparameters(ignore=[...])` to exclude non-serializable objects from checkpoints.
- DataLoaders use `persistent_workers=True` when `num_workers > 0`.
- Use `torch.set_float32_matmul_precision("high")` for TensorFloat32 on Ampere+ GPUs.
