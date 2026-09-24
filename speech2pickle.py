#!/usr/bin/env python3
"""Infer phones from LibriTTS-R speech with ZIPA and serialize them."""

import argparse
import io
import pickle
import re
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from datasets import Audio, load_dataset
from lhotse.features.kaldi.extractors import Fbank, FbankConfig
from panphon import FeatureTable
from pyarrow.parquet import ParquetFile
from tqdm import tqdm


PHONE_RE = re.compile(r"tʃ|dʒ|aɪ|aʊ|eɪ|oʊ|ɔɪ|[^\s](?:[\u0300-\u036f]*)(?:ː)?")


def load_tokens(path):
    return {int(i): token for token, i in (line.split() for line in path.read_text().splitlines())}


def decode(log_probs, lengths, tokens):
    """Apply greedy CTC decoding and return one IPA-phone list per utterance."""
    results = []
    for ids, length in zip(log_probs.argmax(-1), lengths):
        phones, previous = [], -1
        for idx in ids[:length]:
            idx = int(idx)
            if idx and idx != previous and not tokens.get(idx, "").startswith("<"):
                token = tokens.get(idx, "")
                if token != "▁":
                    phones.append(token)
            previous = idx
        results.append(PHONE_RE.findall("".join(phones)))
    return results


def audio_batches(dataset, batch_size, bucket_size):
    buffer = []
    for sidx, sample in enumerate(dataset):
        audio, sr = sf.read(io.BytesIO(sample["audio"]["bytes"]), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(1)
        if sr != 16_000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16_000)
        buffer.append((sidx, sample["speaker_id"], torch.from_numpy(audio)))
        if len(buffer) == bucket_size:
            buffer.sort(key=lambda x: len(x[2]))
            yield from (buffer[i:i + batch_size] for i in range(0, len(buffer), batch_size))
            buffer = []
    if buffer:
        buffer.sort(key=lambda x: len(x[2]))
        yield from (buffer[i:i + batch_size] for i in range(0, len(buffer), batch_size))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("database/libritts_r"))
    parser.add_argument("--model-dir", type=Path, default=Path("pretrained/zipa"))
    parser.add_argument("--output", type=Path, default=Path("data_indices/test.clean.s2p.pkl"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bucket-size", type=int, default=128)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    files = sorted((args.dataset_dir / "test.clean").glob("test.clean-*.parquet"))
    total = sum(ParquetFile(path).metadata.num_rows for path in files)
    dataset = load_dataset("parquet", data_files={"test.clean": list(map(str, files))}, split="test.clean", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False)).select_columns(["audio", "speaker_id"])
    if args.limit:
        total, dataset = min(total, args.limit), dataset.take(args.limit)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    session = ort.InferenceSession(args.model_dir / "model.fp16.onnx", providers=providers)
    tokens, fbank, panphon = load_tokens(args.model_dir / "tokens.txt"), Fbank(FbankConfig(num_filters=80, dither=0.0, snip_edges=False)), FeatureTable()
    data = {}

    with tqdm(total=total, unit="sample") as progress:
        for batch in audio_batches(dataset, args.batch_size, args.bucket_size):
            # ZIPA predicts frame-level IPA tokens from 80-bin filterbank features.
            feats = fbank.extract_batch([x[2] for x in batch], sampling_rate=16_000)
            lengths = np.asarray([len(x) for x in feats], dtype=np.int64)
            padded = torch.nn.utils.rnn.pad_sequence(feats, batch_first=True).numpy()
            log_probs, output_lengths = session.run(None, {"x": padded, "x_lens": lengths})
            for (sidx, spkid, _), phones in zip(batch, decode(log_probs, output_lengths, tokens)):
                # Store the decoded phones and their 24-dimensional PanPhon
                # articulatory representation for downstream RPS work.
                vectors = np.asarray(panphon.word_to_vector_list("".join(phones), numeric=True), dtype=np.int8).reshape(-1, 24)
                data[str(sidx)] = {"spkid": spkid, "phonemes": phones, "articulatory": vectors}
            progress.update(len(batch))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(data)} samples to {args.output} using {session.get_providers()[0]}")


if __name__ == "__main__":
    main()
