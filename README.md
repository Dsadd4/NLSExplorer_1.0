# NLSExplorer

NLSExplorer predicts nuclear localization and highlights candidate nuclear localization signal
(NLS) regions with the interpretable Attention to Key Areas (A2KA) model.

- **Web application:** <http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/>
- **Paper:** [Discovering nuclear localization signal universe through a novel deep learning model with interpretable attention units](https://doi.org/10.1016/j.patter.2025.101262)
- **Local prediction:** sequence, FASTA, or CSV input on CPU, Apple Silicon, or CUDA
- **A2KA:** reusable attention module for other sequence-embedding tasks
- **SCNLS:** dependency-light continuous and discontinuous pattern mining

The maintained interfaces are the `nlsexplorer` and `scnls` commands and the modules under
`src/nlsexplorer/`. The older training and paper-analysis scripts remain available for research
reproduction, but they are not required for normal prediction.

## Installation

NLSExplorer supports Python 3.10–3.12 and PyTorch 2.1 or newer.

Install the PyTorch build appropriate for your CPU or CUDA system by following the
[official PyTorch selector](https://pytorch.org/get-started/locally/), then install this repository:

```bash
git clone https://github.com/Dsadd4/NLSExplorer_1.0.git
cd NLSExplorer_1.0

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[inference]"
```

Conda users can create the supplied environment instead:

```bash
conda env create -f environment.yml
conda activate nlsexplorer
```

## Fastest example: SCNLS

SCNLS does not need the NLSExplorer checkpoint or ESM-1b:

```bash
scnls \
  --sequence MSSAKRRKK \
  --sequence LSSSSKVR \
  --k 3 \
  --max-gap 3 \
  --workers 2
```

Python API:

```python
from nlsexplorer import SCNLS

result = SCNLS(
    ["MSSAKRRKK", "LSSSSKVR", "MTNLP"],
    3,       # maximum component length
    3,       # number of alternating retained/gap components
    2,       # workers; automatically capped at the number of sequences
)

print(result.gapped_patterns)
print(result.continuous_patterns)
```

CSV input is also supported:

```bash
scnls --csv example.csv --sequence-column "Recommended Segment" --format json
```

The historical `python SCNLS.py --mode s|f ...` commands are still accepted.

## Local NLS prediction

Local prediction uses two released models:

1. The lean NLSExplorer A2KA checkpoint, approximately 252 MB on disk.
2. Meta ESM-1b, used to generate the 1,280-dimensional residue embeddings.

The original website checkpoint is approximately 2.88 GB because it contains 656 million
parameters from an embedding layer that was declared but never called by the published forward
pass. The v1.1.0 release checkpoint removes only those unused `*.ll.weight` records. The maintained
loader also accepts the original file, memory-maps it, skips the same records, and strictly checks
every parameter that affects prediction. The active NLSExplorer network contains about 63 million
parameters (approximately 252 MB in Float32), so CPU inference is supported; CUDA is recommended
for higher throughput.

### 1. Download the NLSExplorer checkpoint

```bash
nlsexplorer download-model
nlsexplorer download-esm
```

The downloader writes a `.part` file and resumes interrupted downloads with HTTP Range requests.
By default, the final checkpoint is stored under `~/.cache/nlsexplorer/` (or
`$XDG_CACHE_HOME/nlsexplorer/`).

> **Do not unzip model checkpoints.** The legacy website file `NLS_loc_modeltes` is already a
> complete `torch.save()` checkpoint even though it has no `.pt` extension. `data.pkl`, `data/`,
> and `version` are internal records and cannot be loaded separately.

Inspect any existing download before using it:

```bash
nlsexplorer check-model /path/to/NLS_loc_modeltes
```

### 2. Predict one sequence

```bash
nlsexplorer predict \
  --sequence MSSAKRRKKLSSSSKVR \
  --device auto \
  --download
```

`--download` downloads both missing model files through the resumable downloader. If you omit
`--download` and do not provide `--esm-checkpoint`, `fair-esm` uses its standard PyTorch cache.

The JSON output contains:

- `nuclear_probability` — protein-level model probability;
- `attention` — normalized residue-level attention from the published selective-BAU profile;
- `predicted_regions` — deterministic candidate regions with 1-based inclusive coordinates.

### 3. FASTA or CSV batches

```bash
nlsexplorer predict \
  --fasta proteins.fasta \
  --output predictions.tsv \
  --format tsv \
  --device cuda
```

```bash
nlsexplorer predict \
  --csv proteins.csv \
  --id-column id \
  --sequence-column sequence \
  --output predictions.json
```

ESM-1b accepts at most 1,022 residues. Longer inputs are explicitly reported and truncated to the
first 1,022 residues, matching the published research scripts.

### Python prediction API

```python
from nlsexplorer.predictor import NLSExplorerPredictor

predictor = NLSExplorerPredictor.from_checkpoint(
    "models/NLS_loc_modeltes.pt",
    device="auto",
)

results = predictor.predict(
    ["MSSAKRRKKLSSSSKVR"],
    identifiers=["protein_1"],
)
print(results[0].to_dict())
```

## Accidentally extracted checkpoint

This directory is an accidentally extracted PyTorch checkpoint, not a Lightning checkpoint:

```text
NLS_loc_modeltes/
├── data/
├── data.pkl
└── version
```

Recover a legacy checkpoint without downloading 2.88 GB again:

```bash
nlsexplorer recover-checkpoint \
  Recommendation_system/NLS_loc_modeltes \
  --output models/NLS_loc_modeltes.pt
```

Then verify it:

```bash
nlsexplorer check-model models/NLS_loc_modeltes.pt
```

Optionally create the same lean format distributed in the v1.1.0 release:

```bash
nlsexplorer optimize-checkpoint \
  models/NLS_loc_modeltes.pt \
  --output models/NLSExplorer_A2KA_v1.1.0.pt
```

Do not call `torch.load(.../data.pkl)`. The inner pickle references tensor storages in `data/` and
cannot be deserialized by itself.

## Fully offline inference

Prepare the following on a connected machine and transfer them without extracting:

1. The NLSExplorer checkpoint from `nlsexplorer download-model`.
2. The official `esm1b_t33_650M_UR50S.pt` file.
3. Optionally, `esm1b_t33_650M_UR50S-contact-regression.pt` in the same directory. Contact weights
   are not required for NLSExplorer residue embeddings.
4. A Python environment containing this package, PyTorch, and `fair-esm`.

Run with explicit local files:

```bash
nlsexplorer predict \
  --fasta proteins.fasta \
  --checkpoint /offline/models/NLS_loc_modeltes.pt \
  --esm-checkpoint /offline/models/esm1b_t33_650M_UR50S.pt \
  --device cpu \
  --output predictions.json
```

The local ESM loader explicitly handles the PyTorch 2.6+ `weights_only` default change used by
older official ESM checkpoints.

## Reusable A2KA module

```python
import torch
from nlsexplorer import A2KA

model = A2KA(hidden_dim=512, config=[8, 8, 32])
embedding = torch.randn(5, 100, 512)
probability, layer_attention = model(embedding)

print(probability.shape)       # (5, 1)
print(len(layer_attention))    # 3
```

The compatibility import remains available after installing this repository:

```python
from A2KA import A2KA, SCNLS
```

`hidden_dim` must be divisible by every value in `config`; invalid configurations now fail at
construction time with a direct explanation.

## Development and tests

```bash
python -m pip install -e ".[inference,dev]"
python -m pytest -q
python -m ruff check src tests A2KA.py SCNLS.py
```

The tests use small synthetic checkpoints and models, so CI does not download the 2.88 GB
checkpoint or ESM-1b. They cover installation, CPU inference, strict checkpoint compatibility,
accidental extraction detection and recovery, resumable downloads, CLI behavior, FASTA/CSV I/O,
A2KA, and SCNLS edge cases.

The historical folders `A2KA_train/`, `Recommendation_system/`, and `NLSEXplorer_A2KA/` contain
paper-training and analysis artifacts. They are retained for reproducibility and are not the public
inference interface.

## Citation

```bibtex
@article{li2025nlsexplorer,
  author  = {Yi-Fan Li and Xiaoyong Pan and Hong-Bin Shen},
  title   = {Discovering nuclear localization signal universe through a novel deep learning model with interpretable attention units},
  journal = {Patterns},
  year    = {2025},
  doi     = {10.1016/j.patter.2025.101262},
  url     = {https://www.cell.com/patterns/fulltext/S2666-3899(25)00110-2}
}
```

See [LICENSE](LICENSE) for the repository's academic and non-commercial use terms.
