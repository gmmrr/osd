# OSD

## 1. NvvMix

(TBD)

## 2. OSD

This project is mainly for overlap detection. The primary entry point is `osd.py`, which reads `.wav` files from a folder, runs a local model, and writes one `*_osd.json` file per input.

### 2.1 Environment

Install or refresh the Python environment with:

```bash
uv sync
```

If dependencies change later, run `uv add ...` and then `uv sync` again.

### 2.2 `osd.py`

Use `osd.py` to detect overlap in a directory of `.wav` files:

```bash
uv run python osd.py \
  --input-dir path/to/your/audio \
  --output-dir path/to/your/output \
  --model-path models/segmentation-3.0-ft-train-v5/checkpoints/last.ckpt \
  --onset 0.5 \
  --offset 0.5 \
  --force
```

Common options:

- `--input-dir`: directory containing `.wav` files
- `--output-dir`: directory where `*_osd.json` files are written
- `--model-path`: local model or checkpoint path
- `--onset` / `--offset`: overlap decision thresholds
- `--min-duration-on` / `--min-duration-off`: post-processing parameters
- `--force`: overwrite existing outputs

The latest checkpoint used by this project is:

```bash
models/segmentation-3.0-ft-train-v5/checkpoints/last.ckpt
```

If you have a newer checkpoint, point `--model-path` to that file instead.

### 2.3 Optional tools

`evaluate.py` and `tools/osd_visualization.py` are helper scripts for checking predictions and inspecting results. They are optional and not required for basic overlap detection.

Example evaluation:

```bash
uv run python evaluate.py \
  --ground-truth data/mixed/test \
  --hypothesis out/mix_00000000_osd.json \
  --tolerance 0.0
```

Example visualization:

```bash
uv run python tools/osd_visualization.py \
  --ground-truth data/mixed/test/audio \
  --hypothesis out \
  --no-ground-truth
```
