# Overlap Speech Detection

This project focuses on NVVs-specific overlap speech detection trained on the NvvMix dataset. The model can detect overlaps such as speech + speech, speech + NVVs, and NVVs + NVVs.

## 1. NvvMix

(TBD)

## 2. OSD

The main entry point is `run_osd_drop.py`. It reads `.wav` files from a directory, runs a local model, and produces a `*_osd.wav` file for each input file. The corresponding `*_osd.json` files record the time periods that contain overlap.

### Environment

Create and activate a virtual environment first. For example, use `uv venv` and then `source .venv/bin/activate`.

Then create or update the Python environment with:

```bash
uv sync
```

If the dependencies change later, run `uv add ...` and then `uv sync` again.

### Execution

Run overlap speech detection with the following command:

```bash
uv run python run_osd_drop.py \
  --input-dir path/to/your/audios/root \
  --output-dir path/to/your/output/dir \
  --model-path models/segmentation-3.0-ft-train-v5/checkpoints/last.ckpt \
  --onset 0.52 \
  --offset 0.48 \
  --overlap-pad 50 \
  --force
```

- `--input-dir`: directory containing `.wav` files
- `--output-dir`: directory where `*_osd.wav` files are written
- `--model-path`: local model or checkpoint path
- `--onset` / `--offset`: overlap decision thresholds. The defaults are `0.52` and `0.48`, respectively.
- `--overlap-pad`: extend each detected overlap segment by this many milliseconds on both sides before dropping audio
- `--force`: overwrite existing outputs

The latest checkpoint used by this project is:

```bash
models/segmentation-3.0-ft-train-v5/checkpoints/last.ckpt
```

If you have a newer checkpoint or model, point `--model-path` to that file instead.

### Optional Tools

`evaluate.py`, `tools/osd_visualization.py`, and `tools/demo_visualization.py` are helper scripts for checking predictions and inspecting results. They are optional and not required for basic overlap detection.

#### Evaluation

```bash
uv run python evaluate.py \
  --ground-truth path/to/your/ground/truth \
  --hypothesis path/to/your/output/json \
  --tolerance 0.0
```

This evaluation currently uses a zero-tolerance time boundary, so a very low absolute score is normal. When comparing runs, focus on the relative difference between them.

#### Visualization

Use `tools/osd_visualization.py` to inspect OSD predictions on the original audio:

```bash
uv run python tools/osd_visualization.py \
  --audio-dir path/to/your/audios/root \
  --osd-json path/to/your/output/json
```

The audio directory should contain the original, unchanged `.wav` files. The visualization shows one panel per audio file, with the detected OSD segments and the JSON path for each result.

You can also compare your own ground truth against the OSD predictions. Use `tools/demo_visualization.py` to inspect both:

```bash
uv run python tools/demo_visualization.py \
  --ground-truth path/to/your/ground/truth \
  --osd-json path/to/your/output/json
```

To create your own ground-truth annotation, you need to keep it in the following format:

```text
path/to/your/ground/truth/ (e.g., data/mixed/example/)
├── audio/
│   ├── mix_00000000.wav
│   ├── mix_00000001.wav
│   ├── ...
├── lists/
│   ├── train.lst
│   ├── dev.lst
│   └── test.lst
├── rttm/
│   ├── train.rttm
│   ├── dev.rttm
│   └── test.rttm
├── uem/
│   ├── train.uem
│   ├── dev.uem
│   └── test.uem
└── database.yml
```

- `audio/`: mixed audio files used as the ground-truth inputs
- `lists/`: file lists for each split
- `rttm/`: reference speaker activity annotations
- `uem/`: evaluation region definitions
- `database.yml`: definition of the dataset root and a mapping from each split to its corresponding list, RTTM, and UEM files

For a concrete example, refer to `pipeline/step_5_mix.py` or `data/mixed/example` to see how the ground truth is generated.
