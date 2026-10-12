#!/usr/bin/env python3
"""Run DiaPer inference directly on a directory of WAV files."""
import argparse
from pathlib import Path
import subprocess
import sys
import json
import shutil
import tempfile

import soundfile as sf


def discover_input_files(input_dir: Path) -> list[Path]:
    """Return WAV files from an input directory in stable order."""
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory does not exist: {input_dir}')
    if not input_dir.is_dir():
        raise NotADirectoryError(f'--input-dir must be a directory: {input_dir}')
    return sorted(path.resolve() for path in input_dir.iterdir()
                  if path.is_file() and path.suffix.lower() == '.wav')


def prepare_inference_data(input_files: list[Path], data_dir: Path) -> None:
    """Create the minimal temporary metadata expected by DiaPer's dataset."""
    wav_scp = []
    reco2dur = []
    segments = []
    utt2spk = []
    spk2utt = []
    for audio_path in input_files:
        uri = audio_path.stem
        duration = float(sf.info(str(audio_path)).duration)
        utterance = f'{uri}-inference'
        speaker = f'{uri}-dummy'
        wav_scp.append(f'{uri} {audio_path}')
        reco2dur.append(f'{uri} {duration:.6f}')
        segments.append(f'{utterance} {uri} 0.000000 {duration:.6f}')
        utt2spk.append(f'{utterance} {speaker}')
        spk2utt.append(f'{speaker} {utterance}')
    for name, lines in {
        'wav.scp': wav_scp,
        'reco2dur': reco2dur,
        'segments': segments,
        'utt2spk': utt2spk,
        'spk2utt': spk2utt,
    }.items():
        (data_dir / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def write_json_outputs(output_dir: Path, infer_data_dir: Path) -> None:
    """Convert DiaPer RTTM output to the JSON schema used by SD3 visualization."""
    for rttm_path in list(output_dir.rglob('*.rttm')):
        segments = []
        for line in rttm_path.read_text(encoding='utf-8').splitlines():
            fields = line.split()
            if len(fields) < 8 or fields[0] != 'SPEAKER':
                continue
            start = float(fields[3]); duration = float(fields[4])
            segments.append({'speaker': fields[7], 'start': round(start, 3),
                             'end': round(start + duration, 3), 'duration': round(duration, 3)})
        uri = rttm_path.stem
        wav_scp = infer_data_dir / 'wav.scp'
        if wav_scp.is_file():
            for line in wav_scp.read_text(encoding='utf-8').splitlines():
                fields = line.split(maxsplit=1)
                if len(fields) == 2 and fields[0] == uri:
                    uri = Path(fields[1].split()[-1]).stem
                    break
        payload = {'uri': uri,
                   'num_speakers': len({item['speaker'] for item in segments}),
                   'num_segments': len(segments), 'segments': segments}
        json_path = output_dir / f'{uri}.json'
        json_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        root_rttm = output_dir / f'{uri}.rttm'
        if rttm_path.resolve() != root_rttm.resolve():
            shutil.copyfile(rttm_path, root_rttm)


def parse_args():
    parser = argparse.ArgumentParser(description='Run DiaPer NvvMix diarization and write RTTM/JSON files.')
    parser.add_argument('--input-dir', required=True, help='Directory containing WAV files.')
    parser.add_argument('--model-dir', required=True, help='Directory containing DiaPer checkpoints.')
    parser.add_argument('--epochs', required=True, help='Checkpoint epoch, for example 14 or 10-14.')
    parser.add_argument('--config', default=str(Path(__file__).with_name('infer.yaml')))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files.')
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    input_files = discover_input_files(input_dir)
    if not input_files:
        print(f'⚠️  No WAV files found in: {input_dir}')
        return
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    if args.force:
        for path in output.rglob('*.json'):
            path.unlink()
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise NotADirectoryError(f'--model-dir must be a directory: {model_dir}')
    print(f"🚀 Diarization in '{input_dir}'")
    print(f'   • Files: {len(input_files)}')
    print(f'   • Model: {model_dir}')
    print(f'   • Device: {args.device}')
    with tempfile.TemporaryDirectory(prefix='diaper-inference-') as temporary_dir:
        inference_data = Path(temporary_dir)
        prepare_inference_data(input_files, inference_data)
        command = [sys.executable, str(Path(__file__).parent / 'diaper' / 'infer.py'), '-c', args.config,
                   '--infer-data-dir', str(inference_data), '--models-path', str(model_dir),
                   '--rttms-dir', str(output), '--epochs', args.epochs, '--gpu', '-1']
        subprocess.run(command, check=True)
        write_json_outputs(output, inference_data)
    print('✅ Diarization completed.')
    print(f'Saved: {output}')


if __name__ == '__main__':
    main()
