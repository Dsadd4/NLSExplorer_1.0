# Checkpoint and inference troubleshooting

## Published NLSExplorer checkpoint

The website file named `NLS_loc_modeltes` is one complete `torch.save()` ZIP64 checkpoint. Its
official size is 2,878,755,922 bytes. The missing `.pt` extension does not make it a directory or a
generic archive that users should extract.

The following layout appears only after accidental extraction:

```text
data.pkl
data/
version
```

`data.pkl` is metadata whose persistent tensor references point into `data/`. Loading it alone
produces `A load persistent id instruction was encountered`, which is expected and does not mean
that the checkpoint requires PyTorch Lightning.

Use the maintained commands:

```bash
nlsexplorer download-model
nlsexplorer check-model /path/to/checkpoint
nlsexplorer recover-checkpoint /path/to/extracted-directory --output recovered.pt
nlsexplorer optimize-checkpoint recovered.pt --output NLSExplorer_A2KA_v1.1.0.pt
```

The downloader keeps a `.part` file and uses HTTP Range requests on the next run. Do not add a
second download implementation without preserving resumability.

## Why the lean loader removes `*.ll.weight`

The released state dict has 7,303 tensor storages and 719,053,826 parameters. Every legacy
`Attention` instance declared `nn.Embedding(500, 1280)` as `ll`, but the only attempted use was
commented out in the research code. There are 1,025 such embeddings:

```text
1,025 × 500 × 1,280 = 656,000,000 unused parameters
```

They occupy 2.624 GB and never participate in the published forward pass. The maintained loader:

1. opens the checkpoint with `weights_only=True` and `mmap=True`;
2. removes only keys ending in `.ll.weight`;
3. builds the active model on the meta device;
4. assigns the remaining tensors;
5. rejects every other missing or unexpected key.

Do not broaden the filtering rule. Any mismatch outside `.ll.weight` indicates a different or
damaged model and must remain an error.

## Device handling

Library code must never call `.cuda()` directly or set `CUDA_VISIBLE_DEVICES`. Resolve one device
with `nlsexplorer.device.resolve_device()` and move the NLSExplorer model, ESM model, input tokens,
and residue embeddings to that same device with `.to(device)`.

Supported requests are `auto`, `cpu`, `mps`, `cuda`, and `cuda:N`. An unavailable requested device
must produce an actionable error instead of silently falling back.

## ESM-1b and offline use

Sequence inference also needs `esm1b_t33_650M_UR50S.pt`. Use `nlsexplorer download-esm` for the
same resumable behavior. For an offline machine, transfer the intact files and pass:

```bash
nlsexplorer predict \
  --checkpoint /models/NLS_loc_modeltes.pt \
  --esm-checkpoint /models/esm1b_t33_650M_UR50S.pt \
  --fasta input.fasta
```

`fair-esm` 2.0 predates the PyTorch 2.6 change that made `weights_only=True` the default. Official
ESM checkpoints contain `argparse.Namespace`, so the explicit local ESM path is loaded with
`weights_only=False`. This exception is restricted to the ESM checkpoint the user explicitly
supplies; the NLSExplorer state dict remains on `weights_only=True`.

## Verification

Routine changes do not download multi-gigabyte files in CI. Required local checks are:

```bash
python -m pytest -q
python -m ruff check src tests A2KA.py SCNLS.py
python -m pip wheel --no-deps .
```

Changes to the production model or checkpoint loader additionally require a full integration run
against the official 2.88 GB file and strict confirmation that every active key loads.
