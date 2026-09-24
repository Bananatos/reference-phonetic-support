#!/usr/bin/env python3
"""Serialize LibriTTS-R phonemes and articulatory features."""

import argparse
import pickle
import re
import unicodedata
from pathlib import Path

import numpy as np
from datasets import load_dataset
from panphon import FeatureTable
from phonemizer import phonemize
from pyarrow.parquet import ParquetFile
from tqdm import tqdm

SPLITS = ("test.clean", "train.clean.100")
PHONE_RE = re.compile(r"tʃ|dʒ|aɪ|aʊ|eɪ|oʊ|ɔɪ|[^\s](?:[\u0300-\u036f]*)(?:ː)?")


def phoneme_to_list(phonemes):
    """Split an IPA string while preserving common English diphthongs."""
    return PHONE_RE.findall(phonemes)


def remove_punctuation(text):
    """Remove Unicode punctuation without discarding IPA diacritics."""
    return " ".join(
        "".join(c for c in text if not unicodedata.category(c).startswith("P")).split()
    )


def batched(iterable, batch_size):
    """Yield bounded batches so the streaming dataset stays memory efficient."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("database/libritts_r"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_indices"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--jobs", type=int, default=64)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    files = {s: sorted((args.dataset_dir / s).glob(f"{s}-*.parquet")) for s in SPLITS}
    if any(not paths for paths in files.values()):
        parser.error("LibriTTS-R parquet files not found")
    ds = load_dataset("parquet", data_files={s: list(map(str, paths)) for s, paths in files.items()}, streaming=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = FeatureTable()

    for split in SPLITS:
        total = sum(ParquetFile(p).metadata.num_rows for p in files[split])
        samples = ds[split]
        if args.limit:
            total, samples = min(total, args.limit), samples.take(args.limit)
        data, sidx = {}, 0
        with tqdm(total=total, desc=split, unit="sample") as progress:
            for batch in batched(samples, args.batch_size):
                sequences = phonemize([x["text_original"] for x in batch], backend="espeak", language="en-us", strip=True, preserve_punctuation=True, njobs=args.jobs)
                for sample, sequence in zip(batch, sequences):
                    # PanPhon encodes each reference-transcript phone as a
                    # 24-dimensional articulatory feature vector.
                    sequence = remove_punctuation(sequence)
                    data[str(sidx)] = {
                        "spkid": sample["speaker_id"],
                        "phonemes": phoneme_to_list(sequence),
                        "articulatory": np.asarray(features.word_to_vector_list(sequence, numeric=True), dtype=np.int8),
                    }
                    sidx += 1
                progress.update(len(batch))
        output = args.output_dir / f"{split}.pkl"
        with output.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(data)} samples to {output}")


if __name__ == "__main__":
    main()
