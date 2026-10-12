#!/usr/bin/env python3
"""Convert NvvMix/Kaldi annotations into the Kaldi files consumed by DiaPer."""
from pathlib import Path
import argparse
import shutil
import yaml


def read_lines(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def read_rttm(path):
    path = Path(path)
    file_path = path / 'rttm' if path.is_dir() else path
    result = {}
    if not file_path.exists():
        return result
    for line in file_path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 8 or fields[0].upper() != 'SPEAKER':
            continue
        uri, start, duration, speaker = fields[1], float(fields[3]), float(fields[4]), fields[7]
        result.setdefault(uri, []).append((start, start + duration, speaker))
    return result


def read_uem(path):
    path = Path(path)
    file_path = path / 'uem' if path.is_dir() else path
    result = {}
    if file_path.exists():
        for line in file_path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 4:
                result[fields[0]] = (float(fields[2]), float(fields[3]))
    return result


def discover_sources(input_dir):
    input_dir = Path(input_dir)
    found = []
    for database in sorted(input_dir.rglob('database.yml')):
        root = database.parent
        try:
            meta = yaml.safe_load(database.read_text()) or {}
        except yaml.YAMLError:
            meta = {}
        partition = root.relative_to(input_dir).as_posix()
        lower = partition.lower()
        split = 'test' if '/test' in f'/{lower}' else 'dev' if '/dev' in f'/{lower}' else 'train'
        if 'other' in lower and 'nonverbal_tts' in lower:
            split = 'train'
        if not (root / 'audio').exists() or not (root / 'rttm').exists():
            continue
        found.append((split, partition, root / split, meta))
    return found


def source_records(root):
    base = root.parent if not (root / 'wav.scp').exists() else root
    split = root.name
    wavs = {}
    wav_file = base / 'wav.scp'
    if wav_file.exists():
        wav_lines = read_lines(wav_file)
    else:
        wav_lines = [f'{uri} {base / "audio" / (uri + ".wav")}' for uri in read_lines(base / 'lists' / f'{split}.lst')]
    for line in wav_lines:
        fields = line.split(maxsplit=1)
        if len(fields) == 2:
            wavs[fields[0]] = fields[1]
    uem = read_uem(base / 'uem' / f'{split}.uem')
    rttm = read_rttm(base / 'rttm' / f'{split}.rttm')
    records = []
    for uri, wav in wavs.items():
        if uri not in rttm or uri not in uem:
            continue
        records.append((uri, wav, uem[uri], rttm[uri]))
    return records


def render_split(output, split, sources, max_speakers):
    target = Path(output) / split
    target.mkdir(parents=True, exist_ok=True)
    for name in ('wav.scp', 'segments', 'utt2spk', 'spk2utt', 'reco2dur', 'uem', 'rttm'):
        (target / name).write_text('')
    wav_lines, segment_lines, utt_lines, dur_lines, uem_lines, rttm_lines = [], [], [], [], [], []
    spk_utts = {}
    count = 0
    for partition, root in sources:
        prefix = partition.replace('/', '_') + '_'
        for uri, wav, (uem_start, uem_end), segments in source_records(root):
            new_uri = prefix + uri
            wav_lines.append(f'{new_uri} {wav}')
            dur_lines.append(f'{new_uri} {uem_end:.3f}')
            uem_lines.append(f'{new_uri} 1 {uem_start:.3f} {uem_end:.3f} overlap')
            speakers = sorted({segment[2] for segment in segments})[:max_speakers]
            for index, speaker in enumerate(speakers):
                local_speaker = f'{new_uri}_spk{index}'
                for seg_index, (start, end, current) in enumerate(segments):
                    if current != speaker:
                        continue
                    utt = f'{new_uri}-{seg_index:05d}'
                    utt_lines.append(f'{utt} {local_speaker}')
                    segment_lines.append(f'{utt} {new_uri} {start:.3f} {end:.3f}')
                    spk_utts.setdefault(local_speaker, []).append(utt)
                    rttm_lines.append(f'SPEAKER {new_uri} 1 {start:.3f} {end-start:.3f} <NA> <NA> {local_speaker} <NA> <NA>')
            count += 1
    (target / 'wav.scp').write_text('\n'.join(wav_lines) + ('\n' if wav_lines else ''))
    (target / 'segments').write_text('\n'.join(segment_lines) + ('\n' if segment_lines else ''))
    (target / 'utt2spk').write_text('\n'.join(utt_lines) + ('\n' if utt_lines else ''))
    (target / 'spk2utt').write_text('\n'.join(f'{spk} ' + ' '.join(utts) for spk, utts in sorted(spk_utts.items())) + ('\n' if spk_utts else ''))
    (target / 'reco2dur').write_text('\n'.join(dur_lines) + ('\n' if dur_lines else ''))
    (target / 'uem').write_text('\n'.join(uem_lines) + ('\n' if uem_lines else ''))
    (target / 'rttm').write_text('\n'.join(rttm_lines) + ('\n' if rttm_lines else ''))
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-speakers', type=int, default=3)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and args.force:
        shutil.rmtree(output)
    sources = discover_sources(args.input_dir)
    print(f'🚀 Prepare DiaPer Kaldi data from {args.input_dir}')
    for split in ('train', 'dev', 'test'):
        selected = [(part, root) for current, part, root, _ in sources if current == split]
        print(f'  {split}: {render_split(output, split, selected, args.max_speakers)} recordings')
    print(f'✅ Wrote Kaldi data to {output}')


if __name__ == '__main__':
    main()
