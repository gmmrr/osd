#!/usr/bin/env python3
"""Evaluate DiaPer RTTM hypotheses against a Kaldi reference directory."""
from pathlib import Path
import argparse
from prepare_kaldi import read_rttm, read_uem


def score(reference, hypothesis, uem):
    refs = reference.get('segments', [])
    hyps = hypothesis.get('segments', [])
    boundaries = sorted({uem[0], uem[1]} | {x for s, e, _ in refs + hyps for x in (s, e)})
    miss = fa = conf = total = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        if right <= uem[0] or left >= uem[1]: continue
        mid = (left + right) / 2
        r = {speaker for start, end, speaker in refs if start <= mid < end}
        h = {speaker for start, end, speaker in hyps if start <= mid < end}
        duration = right - left
        total += duration * len(r)
        miss += duration * max(0, len(r) - len(h))
        fa += duration * max(0, len(h) - len(r))
        conf += duration * min(len(r), len(h)) if r != h else 0
    return total, miss, fa, conf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--hypothesis-dir', required=True)
    args = parser.parse_args()
    ref = read_rttm(Path(args.reference_dir) / 'rttm')
    uem = read_uem(Path(args.reference_dir) / 'uem')
    total = miss = fa = conf = 0.0
    for path in sorted(Path(args.hypothesis_dir).glob('*.rttm')):
        t, m, f, c = score({'segments': ref.get(path.stem, [])}, {'segments': read_rttm(path).get(path.stem, [])}, uem.get(path.stem, (0, 0)))
        total += t; miss += m; fa += f; conf += c
    denom = max(total, 1e-9)
    print(f'📊 DER: {(miss + fa + conf) / denom * 100:.2f}% (miss {miss / denom * 100:.2f}%, false alarm {fa / denom * 100:.2f}%, confusion {conf / denom * 100:.2f}%)')


if __name__ == '__main__':
    main()
