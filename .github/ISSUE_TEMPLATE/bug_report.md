---
name: Bug report
about: Report a reproducible NLSExplorer installation, checkpoint, device, or prediction problem
title: ''
labels: 'bug'
assignees: ''

---

## What happened?

Describe the problem and the result you expected.

## Minimal reproduction

Provide the exact command or shortest Python example that reproduces the problem.

```bash
# command
```

```text
# complete traceback or terminal output
```

## Input

- Input type: raw sequence / FASTA / CSV / precomputed embedding / other
- Number of sequences:
- Longest sequence length:
- Minimal non-sensitive example, if possible:

## Model checkpoint

- Checkpoint source:
- Checkpoint path:
- File size:
- Is the checkpoint still one file, or was it extracted into `data.pkl`, `data/`, and `version`?

`NLS_loc_modeltes` is a complete PyTorch checkpoint file even though it has no `.pt`
extension. Do not extract it or load its internal `data.pkl` directly.

## Environment

Run and paste the output of:

```bash
python --version
python -m pip show nlsexplorer torch fair-esm numpy
python - <<'PY'
import platform
import torch

print("platform:", platform.platform())
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_0:", torch.cuda.get_device_name(0))
PY
```

- Installation method: pip / conda / source checkout / other
- Requested device: auto / cpu / cuda / cuda:N
- Online or offline machine:
- If offline, how were the repository, dependencies, ESM weights, and NLSExplorer
  checkpoint transferred?

## Additional context

Include anything else that changes reproduction, such as available RAM/VRAM, proxy
settings, or whether the same command works on another machine.
