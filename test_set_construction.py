#!/usr/bin/env python3
"""Build phoneme-length-controlled LibriTTS-R RPS test pairs.

The two inputs may be self-contained metadata CSV files.  For compatibility with
``sample2csv.py``, the reference CSV may omit speaker_id; in that case speaker
IDs are read (without decoding audio) from the test.clean parquet files and
joined by sentence_index/row order.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tqdm import tqdm


NUM_FIXED_REFS_PER_SPK = 10
NUM_TARGETS_PER_REF = 10
NUM_FIXED_TGTS_PER_SPK = 10
NUM_REFS_PER_TARGET = 10
TARGET_LENGTH_TOLERANCE = 2
REF_LENGTH_TOLERANCE = 2
MAX_TARGET_LENGTH_TOLERANCE = 5
MAX_REF_LENGTH_TOLERANCE = 5
SEED = 42
STRICT_RPS = True
RPS_BINS = tuple((i / 10, (i + 1) / 10) for i in range(10))

REF_SPLIT = "test.clean"
TGT_SPLIT = "train.clean.100"
PHONE_RE = re.compile(r"tʃ|dʒ|aɪ|aʊ|eɪ|oʊ|ɔɪ|[^\s](?:[\u0300-\u036f]*)(?:ː)?")


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    speaker_id: str | None
    split: str
    phonemes: str
    length: int


@dataclass
class GroupResult:
    rows: list[dict]
    speaker_id: str
    fixed_id: str
    tolerance: int
    candidate_count: int
    achievable_rps_min: float | None
    achievable_rps_max: float | None


def phoneme_to_list(phonemes: str) -> list[str]:
    return PHONE_RE.findall(phonemes)


def phoneme_length(phonemes: str) -> int:
    return len(phoneme_to_list(phonemes))


def calculate_rps(ref_phonemes: str, target_phonemes: str, strict: bool = True) -> float:
    """Return the fraction of unique target phones supported by the reference."""
    def phone_set(value: str) -> set[str]:
        phones = phoneme_to_list(value)
        return set(phones if strict else (p.replace("ː", "") for p in phones))

    p_r, p_t = map(phone_set, (ref_phonemes, target_phonemes))
    if not p_t:
        raise ValueError("target phoneme sequence is empty")
    return len(p_r & p_t) / len(p_t)


def _first(row: dict[str, str], names: Sequence[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _parquet_speakers(dataset_dir: Path, split: str) -> list[str]:
    paths = sorted((dataset_dir / split).glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no parquet files found under {dataset_dir / split}")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "reference CSV has no speaker_id; install pyarrow or provide a "
            "self-contained metadata CSV"
        ) from exc
    speakers: list[str] = []
    for path in paths:
        table = pq.read_table(path, columns=["speaker_id"])
        speakers.extend(str(x) for x in table.column("speaker_id").to_pylist())
    return speakers


def load_metadata(
    path: Path,
    expected_split: str,
    dataset_dir: Path | None = None,
    require_speaker: bool = False,
) -> list[Utterance]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError(f"metadata is empty: {path}")

    fallback_speakers: list[str] | None = None
    if require_speaker and not any(
        _first(r, ("speaker_id", "speaker", "spk_id", "spkid")) for r in raw
    ):
        if dataset_dir is None:
            raise ValueError(f"{path} has no speaker_id column")
        fallback_speakers = _parquet_speakers(dataset_dir, expected_split)
        if len(fallback_speakers) != len(raw):
            raise ValueError(
                f"CSV/parquet row mismatch for {expected_split}: "
                f"{len(raw)} != {len(fallback_speakers)}"
            )

    items: list[Utterance] = []
    rejected = 0
    for row_number, row in enumerate(raw):
        utt_id = _first(
            row, ("utt_id", "ref_id", "tgt_id", "sentence_index", "sidx", "id")
        )
        phonemes = _first(row, ("phonemes", "phoneme", "phonmes", "phones"))
        split = _first(row, ("split",)) or expected_split
        speaker = _first(row, ("speaker_id", "speaker", "spk_id", "spkid"))
        if speaker is None and fallback_speakers is not None:
            speaker = fallback_speakers[row_number]
        if utt_id is None:
            # LibriTTS-R generation/evaluation scripts use split-local row indices.
            utt_id = str(row_number)
        if split != expected_split:
            raise ValueError(f"unexpected split {split!r} in {path}; expected {expected_split!r}")
        if phonemes is None or phoneme_length(phonemes) == 0:
            rejected += 1
            continue
        if require_speaker and speaker is None:
            raise ValueError(f"missing speaker_id at row {row_number + 2} in {path}")
        items.append(Utterance(utt_id, speaker, split, phonemes, phoneme_length(phonemes)))

    ids = [item.utt_id for item in items]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1][:5]
        raise ValueError(f"duplicate utterance IDs in {path}: {duplicates}")
    if rejected:
        print(f"Filtered {rejected} empty/unparseable phoneme rows from {path}")
    if not items:
        raise ValueError(f"no valid phoneme rows in {path}")
    return items


def build_target_pool(items: Sequence[Utterance]) -> dict[int, list[Utterance]]:
    result: dict[int, list[Utterance]] = defaultdict(list)
    for item in items:
        result[item.length].append(item)
    return dict(result)


def build_speaker_reference_pool(items: Sequence[Utterance]) -> dict[str, list[Utterance]]:
    result: dict[str, list[Utterance]] = defaultdict(list)
    for item in items:
        assert item.speaker_id is not None
        result[item.speaker_id].append(item)
    return dict(result)


def _stratified(items: Sequence[Utterance], count: int, rng: random.Random) -> list[Utterance]:
    """One deterministic seeded draw from each equal-position length stratum."""
    ordered = sorted(items, key=lambda x: (x.length, x.utt_id))
    if len(ordered) <= count:
        return ordered
    selected = []
    for i in range(count):
        lo = math.floor(i * len(ordered) / count)
        hi = math.floor((i + 1) * len(ordered) / count)
        selected.append(ordered[rng.randrange(lo, max(lo + 1, hi))])
    return selected


def select_fixed_references(
    refs: Sequence[Utterance], count: int, rng: random.Random
) -> list[Utterance]:
    return _stratified(refs, count, rng)


def select_fixed_targets(
    targets: Sequence[Utterance], count: int, rng: random.Random
) -> list[Utterance]:
    return _stratified(targets, count, rng)


def _rps_bin(value: float) -> int | None:
    for index, (low, high) in enumerate(RPS_BINS):
        if low <= value < high or (index == len(RPS_BINS) - 1 and value == high):
            return index
    return None


def _select_diverse(
    scored: Sequence[tuple[Utterance, float]], count: int, rng: random.Random
) -> list[tuple[Utterance, float]]:
    """Cover bins first, then greedily maximize distance from selected RPS values."""
    tie_order = {item.utt_id: rng.random() for item, _ in scored}
    by_bin: dict[int, list[tuple[Utterance, float]]] = defaultdict(list)
    for pair in scored:
        index = _rps_bin(pair[1])
        assert index is not None
        by_bin[index].append(pair)
    chosen: list[tuple[Utterance, float]] = []
    for index in sorted(by_bin):
        midpoint = sum(RPS_BINS[index]) / 2
        chosen.append(min(by_bin[index], key=lambda p: (abs(p[1] - midpoint), tie_order[p[0].utt_id])))
    if len(chosen) > count:
        # This only matters with custom configurations having more bins than pairs.
        chosen = _farthest(chosen, count, tie_order)
    used = {item.utt_id for item, _ in chosen}
    remaining = [pair for pair in scored if pair[0].utt_id not in used]
    while len(chosen) < count and remaining:
        if chosen:
            key = lambda p: (min(abs(p[1] - q[1]) for q in chosen), -tie_order[p[0].utt_id])
        else:
            key = lambda p: (0.0, -tie_order[p[0].utt_id])
        pick = max(remaining, key=key)
        chosen.append(pick)
        remaining.remove(pick)
    return sorted(chosen, key=lambda p: (p[1], p[0].utt_id))


def _farthest(
    values: Sequence[tuple[Utterance, float]], count: int, tie_order: dict[str, float]
) -> list[tuple[Utterance, float]]:
    ordered = sorted(values, key=lambda p: (p[1], tie_order[p[0].utt_id]))
    chosen = [ordered[0]]
    if count > 1:
        chosen.append(ordered[-1])
    remaining = [x for x in ordered if x not in chosen]
    while len(chosen) < count:
        pick = max(
            remaining,
            key=lambda p: (min(abs(p[1] - q[1]) for q in chosen), -tie_order[p[0].utt_id]),
        )
        chosen.append(pick)
        remaining.remove(pick)
    return chosen


def _best_window(
    candidates: Sequence[Utterance],
    score_against: Utterance,
    fixed_is_ref: bool,
    needed: int,
    initial_tolerance: int,
    max_tolerance: int,
) -> tuple[list[tuple[Utterance, float]], int]:
    # RPS depends only on the two phone sets, so compute it once per pair.
    by_length: dict[int, list[tuple[Utterance, float]]] = defaultdict(list)
    for item in candidates:
        value = (
            calculate_rps(score_against.phonemes, item.phonemes, STRICT_RPS)
            if fixed_is_ref
            else calculate_rps(item.phonemes, score_against.phonemes, STRICT_RPS)
        )
        by_length[item.length].append((item, value))
    lengths = sorted(by_length)
    last: list[tuple[Utterance, float]] = []
    for tolerance in range(initial_tolerance, max_tolerance + 1):
        options = []
        for anchor in lengths:
            scored = [pair for length in range(anchor - tolerance, anchor + tolerance + 1)
                      for pair in by_length.get(length, ())]
            last = scored if len(scored) > len(last) else last
            if len(scored) >= needed:
                rps_values = [value for _, value in scored]
                actual_range = max(x.length for x, _ in scored) - min(x.length for x, _ in scored)
                coverage = len({_rps_bin(value) for value in rps_values})
                # Prefer tighter length matching, then broader RPS coverage.
                quality = (-actual_range, coverage, max(rps_values) - min(rps_values), -len(scored), -anchor)
                options.append((quality, scored))
        if options:
            return max(options, key=lambda x: x[0])[1], tolerance
    return last, max_tolerance


def select_targets_for_fixed_ref(
    speaker: str, ref: Utterance, targets: Sequence[Utterance], rng: random.Random
) -> GroupResult:
    scored, tolerance = _best_window(
        targets, ref, True, NUM_TARGETS_PER_REF,
        TARGET_LENGTH_TOLERANCE, MAX_TARGET_LENGTH_TOLERANCE,
    )
    selected = _select_diverse(scored, NUM_TARGETS_PER_REF, rng) if scored else []
    values = [value for _, value in scored]
    rows = [{"ref_id": ref.utt_id, "tgt_id": tgt.utt_id, "rps": value,
             "speaker_id": speaker, "ref_len": ref.length, "tgt_len": tgt.length}
            for tgt, value in selected]
    return GroupResult(rows, speaker, ref.utt_id, tolerance, len(scored),
                       min(values) if values else None, max(values) if values else None)


def select_refs_for_fixed_target(
    speaker: str, target: Utterance, refs: Sequence[Utterance], rng: random.Random
) -> GroupResult:
    scored, tolerance = _best_window(
        refs, target, False, NUM_REFS_PER_TARGET,
        REF_LENGTH_TOLERANCE, MAX_REF_LENGTH_TOLERANCE,
    )
    selected = _select_diverse(scored, NUM_REFS_PER_TARGET, rng) if scored else []
    values = [value for _, value in scored]
    rows = [{"ref_id": ref.utt_id, "tgt_id": target.utt_id, "rps": value,
             "speaker_id": speaker, "ref_len": ref.length, "tgt_len": target.length}
            for ref, value in selected]
    return GroupResult(rows, speaker, target.utt_id, tolerance, len(scored),
                       min(values) if values else None, max(values) if values else None)


def build_fixed_ref_dataset(
    speaker_refs: dict[str, list[Utterance]], targets: Sequence[Utterance], seed: int
) -> list[GroupResult]:
    results = []
    total = sum(min(len(refs), NUM_FIXED_REFS_PER_SPK) for refs in speaker_refs.values())
    with tqdm(total=total, desc="Sampling fixed-reference groups", unit="group") as progress:
        for speaker in sorted(speaker_refs):
            rng = random.Random(f"{seed}:fixed-ref:{speaker}")
            for ref in select_fixed_references(speaker_refs[speaker], NUM_FIXED_REFS_PER_SPK, rng):
                results.append(select_targets_for_fixed_ref(speaker, ref, targets, rng))
                progress.update()
    return results


def build_fixed_tgt_dataset(
    speaker_refs: dict[str, list[Utterance]], targets: Sequence[Utterance], seed: int
) -> list[GroupResult]:
    results = []
    targets_per_speaker = min(len(targets), NUM_FIXED_TGTS_PER_SPK)
    total = len(speaker_refs) * targets_per_speaker
    with tqdm(total=total, desc="Sampling fixed-target groups", unit="group") as progress:
        for speaker in sorted(speaker_refs):
            rng = random.Random(f"{seed}:fixed-tgt:{speaker}")
            for target in select_fixed_targets(targets, NUM_FIXED_TGTS_PER_SPK, rng):
                results.append(select_refs_for_fixed_target(speaker, target, speaker_refs[speaker], rng))
                progress.update()
    return results


def validate_dataset(
    results: Sequence[GroupResult], refs: Sequence[Utterance], targets: Sequence[Utterance],
    fixed_ref: bool,
) -> None:
    rows = [row for result in results for row in result.rows]
    ref_ids, tgt_ids = {x.utt_id for x in refs}, {x.utt_id for x in targets}
    assert all(row["ref_id"] in ref_ids for row in rows)
    assert all(row["tgt_id"] in tgt_ids for row in rows)
    assert all(math.isfinite(row["rps"]) and 0 <= row["rps"] <= 1 for row in rows)
    pairs = [(row["ref_id"], row["tgt_id"]) for row in rows]
    assert len(pairs) == len(set(pairs)), "duplicate ref_id/tgt_id pairs"
    expected = NUM_TARGETS_PER_REF if fixed_ref else NUM_REFS_PER_TARGET
    for result in results:
        assert len({(r["tgt_id"] if fixed_ref else r["ref_id"]) for r in result.rows}) == len(result.rows)
        lengths = [r["tgt_len" if fixed_ref else "ref_len"] for r in result.rows]
        if lengths:
            # Materialize all requested within-group checks, including for an
            # incomplete group, so malformed length data cannot pass silently.
            length_stats = {
                "min": min(lengths),
                "max": max(lengths),
                "mean": statistics.mean(lengths),
                "std": statistics.pstdev(lengths),
                "range": max(lengths) - min(lengths),
            }
            assert all(math.isfinite(value) for value in length_stats.values())
        if len(result.rows) == expected:
            assert max(lengths) - min(lengths) <= 2 * result.tolerance


def _summary(values: Sequence[float]) -> str:
    if not values:
        return "n=0"
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return (f"n={len(values)}, min={min(values):.4f}, max={max(values):.4f}, "
            f"mean={statistics.mean(values):.4f}, median={statistics.median(values):.4f}, "
            f"std={std:.4f}")


def print_diagnostics(name: str, results: Sequence[GroupResult], expected: int, initial: int) -> None:
    rows = [row for result in results for row in result.rows]
    incomplete = [result for result in results if len(result.rows) != expected]
    length_key = "tgt_len" if name == "fixed-ref" else "ref_len"
    ranges = [max(r[length_key] for r in result.rows) - min(r[length_key] for r in result.rows)
              for result in results if result.rows]
    print(f"\n{name} diagnostics")
    print(f"  groups={len(results)}, pairs={len(rows)}, incomplete_groups={len(incomplete)}")
    print(f"  RPS: {_summary([r['rps'] for r in rows])}")
    print(f"  within-group phoneme-length range: {_summary(ranges)}")
    print(f"  groups at initial tolerance={sum(r.tolerance == initial for r in results)}, "
          f"relaxed={sum(r.tolerance > initial for r in results)}, "
          f"maximum tolerance used={max((r.tolerance for r in results), default=0)}")
    for result in incomplete:
        lo = "n/a" if result.achievable_rps_min is None else f"{result.achievable_rps_min:.4f}"
        hi = "n/a" if result.achievable_rps_max is None else f"{result.achievable_rps_max:.4f}"
        print(f"  INCOMPLETE speaker_id={result.speaker_id} fixed_id={result.fixed_id} "
              f"selected={len(result.rows)}/{expected} candidates={result.candidate_count} "
              f"RPS_range=[{lo}, {hi}] tolerance={result.tolerance}")


def _write(path: Path, results: Sequence[GroupResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ref_id", "tgt_id", "rps"])
        writer.writeheader()
        for result in results:
            for row in result.rows:
                writer.writerow({"ref_id": row["ref_id"], "tgt_id": row["tgt_id"],
                                 "rps": f"{row['rps']:.10f}"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref-metadata", type=Path, default=Path("data_indices/test-clean.csv"))
    parser.add_argument("--tgt-metadata", type=Path, default=Path("data_indices/train-clean-100.csv"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("database/libritts_r"))
    parser.add_argument("--output-dir", type=Path, default=Path("data_indices"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--allow-nonstandard-speaker-count", action="store_true",
                        help="permit a test.clean speaker count other than 39 (for tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = load_metadata(args.ref_metadata, REF_SPLIT, args.dataset_dir, require_speaker=True)
    targets = load_metadata(args.tgt_metadata, TGT_SPLIT)
    speaker_refs = build_speaker_reference_pool(refs)
    if len(speaker_refs) != 39 and not args.allow_nonstandard_speaker_count:
        raise ValueError(f"test.clean must contain 39 speakers, found {len(speaker_refs)}")
    print(f"Number of speakers: {len(speaker_refs)}")

    fixed_ref = build_fixed_ref_dataset(speaker_refs, targets, args.seed)
    fixed_tgt = build_fixed_tgt_dataset(speaker_refs, targets, args.seed)
    validate_dataset(fixed_ref, refs, targets, fixed_ref=True)
    validate_dataset(fixed_tgt, refs, targets, fixed_ref=False)

    fixed_ref_path = args.output_dir / "test.clean_x_train.clean.100_fixed_ref.csv"
    fixed_tgt_path = args.output_dir / "test.clean_x_train.clean.100_fixed_tgt.csv"
    _write(fixed_ref_path, fixed_ref)
    _write(fixed_tgt_path, fixed_tgt)
    print_diagnostics("fixed-ref", fixed_ref, NUM_TARGETS_PER_REF, TARGET_LENGTH_TOLERANCE)
    print_diagnostics("fixed-tgt", fixed_tgt, NUM_REFS_PER_TARGET, REF_LENGTH_TOLERANCE)
    print(f"\nWrote {fixed_ref_path}\nWrote {fixed_tgt_path}")


if __name__ == "__main__":
    main()
