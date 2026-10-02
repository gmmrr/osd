# Independent Python environments

Run every command below from the repository root on macOS Apple Silicon. The four environments use CPython 3.12.9. `envs/diarization/` is separate and is not changed by this setup.

The requirement files reflect imports in `pipeline/`, `osd/`, the two native `osdc/` training paths, CornellOSDC, and the shared visualization tools. Their direct dependencies are pinned to versions used in the former root environment. Transitive dependencies are resolved by `uv pip install`.

| Code | Environment | Requirements |
| --- | --- | --- |
| `pipeline/`, `config/`, NvvMix visualization | `envs/nvvmix` | `requirements/nvvmix.txt` |
| `osd/`, OSD visualization | `envs/osd` | `requirements/osd.txt` |
| `osdc/train.py`, `osdc/alternatives/train.py`, native OSDC evaluation and visualization | `envs/osdc/pyannote3` | `requirements/osdc-pyannote3.txt` |
| `osdc/external/CornellOSDC/`, Cornell OSDC visualization | `envs/osdc/cornell` | `requirements/osdc-cornell.txt` |

All four environments include `matplotlib`, `pyqtgraph`, `PySide6`, `numpy`, `scipy`, `soundfile`, and `tensorboardX` for the shared visualization scripts and log viewer. `PySide6` also provides Qt Multimedia, which `tools/utils/visualization.py` imports.

## Create and install

If your current shell has the former root `.venv` activated, run `deactivate` first. The root `.venv` has been removed. In VS Code, select the relevant `envs/.../bin/python` interpreter for the files you are working on.

```bash
uv python install 3.12.9

uv venv --python 3.12.9 envs/nvvmix
uv venv --python 3.12.9 envs/osd
uv venv --python 3.12.9 envs/osdc/pyannote3
uv venv --python 3.12.9 envs/osdc/cornell

uv pip install --python envs/nvvmix/bin/python -r requirements/nvvmix.txt
uv pip install --python envs/osd/bin/python -r requirements/osd.txt
uv pip install --python envs/osdc/pyannote3/bin/python -r requirements/osdc-pyannote3.txt
uv pip install --python envs/osdc/cornell/bin/python -r requirements/osdc-cornell.txt
```

`envs/` is ignored by Git. Requirements and this guide are tracked. To rebuild an existing environment, use `uv venv --clear --python 3.12.9 envs/<name>` and reinstall its requirements. Do not run `uv sync` or bare `uv run python ...` from the root: those use the root project's `.venv`.

## Run from the repository root

The `--no-project` option prevents root `pyproject.toml` discovery. `--python` selects the desired environment. Use this syntax for all scripts:

```bash
uv run --no-project --python envs/nvvmix/bin/python python pipeline/step_5_mix.py --help
uv run --no-project --python envs/osd/bin/python python osd/train.py --help
uv run --no-project --python envs/osdc/pyannote3/bin/python python osdc/train.py --help
uv run --no-project --python envs/osdc/pyannote3/bin/python python osdc/alternatives/train.py --help
uv run --no-project --python envs/osdc/cornell/bin/python python osdc/external/CornellOSDC/egs/AMI/local/train.py --help
```

For Cornell training with the local AMI config and an existing checkpoint:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 caffeinate -im \
  uv run --no-project --python envs/osdc/cornell/bin/python python \
  osdc/external/CornellOSDC/egs/AMI/local/train.py \
  osdc/external/CornellOSDC/egs/AMI/conf/train.local.yml \
  osdc/external/CornellOSDC/models/cornell-based-train-v1 \
  --resume osdc/external/CornellOSDC/models/cornell-based-train-v1/checkpoints/last.ckpt
```

Examples of visualization in each environment:

```bash
uv run --no-project --python envs/nvvmix/bin/python python tools/mix_visualization.py --help
uv run --no-project --python envs/osd/bin/python python tools/osd_visualization.py --help
uv run --no-project --python envs/osdc/pyannote3/bin/python python tools/osdc_visualization.py --help
uv run --no-project --python envs/osdc/cornell/bin/python python tools/osdc_visualization.py --help
```

`tools/osdc_visualization.py` reads JSON outputs, so either OSDC environment can run it. To inspect training logs, run `tools/view_checkpoint_logs.py` with the relevant environment.

## Verify isolation

```bash
uv run --no-project --python envs/nvvmix/bin/python python -c 'import sys, numpy, soundfile, pyloudnorm, silero_vad, pyqtgraph, PySide6; print(sys.executable)'
uv run --no-project --python envs/osd/bin/python python -c 'import sys, torch, lightning, pyannote.audio, pyannote.database, pyannote.metrics, pyqtgraph, PySide6; print(sys.executable)'
uv run --no-project --python envs/osdc/pyannote3/bin/python python -c 'import sys, torch, lightning, pyannote.audio, pyqtgraph, PySide6; print(sys.executable)'
uv run --no-project --python envs/osdc/cornell/bin/python python -c 'import sys, torch, torchaudio, asteroid, intervals, yaml, pyqtgraph, PySide6; print(sys.executable)'
```

For Apple Silicon, check MPS separately in each training environment:

```bash
uv run --no-project --python envs/osd/bin/python python -c 'import torch; print(torch.backends.mps.is_available())'
uv run --no-project --python envs/osdc/pyannote3/bin/python python -c 'import torch; print(torch.backends.mps.is_available())'
uv run --no-project --python envs/osdc/cornell/bin/python python -c 'import torch; print(torch.backends.mps.is_available())'
```
