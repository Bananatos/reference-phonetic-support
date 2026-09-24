# GitHub upload checklist

Repository name: **Reference Phonetic Support (RPS)**

## Upload now

### Scripts

- [x] `sample2pickle.py` — derive reference-transcript IPA phones and PanPhon articulatory features from LibriTTS-R parquet files.
- [x] `speech2pickle.py` — infer reference-speech IPA phones with ZIPA and save PanPhon articulatory features.
- [x] `test_set_construction.py` — compute RPS and construct the fixed-reference and fixed-target TTS input pairs.
- [x] `append_ngram_rps.py` — reproducibly append 1-, 2-, 3-, 5-, and 7-gram RPS columns.

### Generated data

- [x] `data_indices/test.clean_x_train.clean.100_fixed_ref.csv`
- [x] `data_indices/test.clean_x_train.clean.100_fixed_tgt.csv`
- [x] `data_indices/test.clean.pkl`
- [x] `data_indices/train.clean.100.pkl`

### Documentation and environment

- [x] `README.md`
- [x] `requirements.txt`
- [x] `github_upload.md`
- [x] `CITATION.cff` — marks the ICASSP 2027 manuscript as submitted, not published.

### External components

- [x] Document ZIPA package dependencies and the required `anyspeech/zipa-large-crctc-300k` files.
- [x] Point users to model-specific download, inference, and evaluation instructions under `TTS/`.
- [x] Represent TTS implementations with upstream links; do not publish the local symbolic link or vendor separately licensed repositories.

## Decide before public release

- [ ] Add the paper/preprint URL when one becomes available.
- [ ] Add the repository license (`LICENSE`) and confirm compatibility with redistributed artifacts and any TTS code.
- [ ] Document the exact TTS checkpoints, decoding settings, output filename convention, and speaker-verification checkpoint used in the submitted experiments.
- [ ] Add or release the project-specific TTS generation and speaker-similarity adapters if they are intended to be reproducible source code.
- [ ] Record the final CUDA, PyTorch, and ONNX Runtime versions used for ZIPA inference.
- [ ] Verify permission to redistribute the derived pickle files; use Git LFS or release assets if appropriate.
- [ ] Run a clean-environment smoke test on a small `--limit` subset.
- [x] Initialize Git, review every staged path, and exclude unrelated experiments, checkpoints, datasets, caches, local symbolic links, and credentials.

## Validation audit

- [x] Public scripts and released CSV headers consistently use `RPS`/`rps`.
- [x] The n-gram implementation uses `NGRAMS = (1, 2, 3, 5, 7)`.
- [x] Both released pair CSVs contain a `7-gram` column.
- [x] Existing numeric RPS values were preserved.
