#!/usr/bin/env python3
"""Append phoneme n-gram RPS columns to the two controlled pair CSVs."""

import argparse
import csv
import re
from pathlib import Path

from tqdm import tqdm


PHONE_RE = re.compile(r"tʃ|dʒ|aɪ|aʊ|eɪ|oʊ|ɔɪ|[^\s](?:[\u0300-\u036f]*)(?:ː)?")
NGRAMS = (1, 2, 3, 5, 7)
PAIR_FILES = (
    "test.clean_x_train.clean.100_fixed_ref.csv",
    "test.clean_x_train.clean.100_fixed_tgt.csv",
)


def phoneme_to_list(phonemes):
    return PHONE_RE.findall(phonemes)


def rps_ngram(num, ref_seq, tgt_seq):
    """Fraction of distinct target n-grams found in the reference sequence."""
    ref = set(zip(*(ref_seq[i:] for i in range(num))))
    tgt = set(zip(*(tgt_seq[i:] for i in range(num))))
    return len(ref & tgt) / len(tgt) if tgt else 0.0


def load_phonemes(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        return {
            int(row["sidx"]): phoneme_to_list(row["phonmes"])
            for row in rows
        }


def append_columns(path, refs, tgts):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), list(reader.fieldnames or ())

    columns = [f"{n}-gram" for n in NGRAMS]
    for row in tqdm(rows, desc=path.name, unit="pair"):
        ref_seq = refs[int(row["ref_id"])]
        tgt_seq = tgts[int(row["tgt_id"])]
        row.update({column: f"{rps_ngram(n, ref_seq, tgt_seq):.10f}"
                    for n, column in zip(NGRAMS, columns)})

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + [c for c in columns if c not in fields])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data_indices"))
    args = parser.parse_args()

    refs = load_phonemes(args.data_dir / "test-clean.csv")
    tgts = load_phonemes(args.data_dir / "train-clean-100.csv")
    for name in PAIR_FILES:
        append_columns(args.data_dir / name, refs, tgts)


if __name__ == "__main__":
    main()
