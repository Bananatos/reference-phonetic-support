# Reference Phonetic Support (RPS)

This repository contains the core scripts and intermediate indices used to study **Reference Phonetic Support (RPS)** in reference-conditioned text-to-speech (TTS). It builds controlled LibriTTS-R reference/target pairs, generates IPA and articulatory representations, and defines the evaluation workflow for measuring speaker similarity under different RPS conditions.

## Paper

**Rethinking Speaker Similarity Evaluation for Zero-Shot TTS: Phonetic-Content Dependence and Reference Phonetic Support**

Jiabei He and Yong Qin\*  
College of Computer Science, Nankai University, Tianjin 300350, China  
Email: `hejiabei@mail.nankai.edu.cn`; `qinyong@nankai.edu.cn`

\* Corresponding author: Yong Qin. The manuscript is currently under review for ICASSP 2027 and has not yet been published.

This work was supported by the National Science and Technology Major Project under Grant 2022ZD0116307 and the National Natural Science Foundation of China under Grant 62271270.

## RPS definition

For reference phones $P_r$ and target-text phones $P_t$, the scripts compute

$$
\mathrm{RPS}(r,t)=\frac{|\operatorname{set}(P_r)\cap\operatorname{set}(P_t)|}{|\operatorname{set}(P_t)|}.
$$

RPS is directional: it measures the fraction of unique target phones supported by the reference. The default strict mode treats long and short IPA phones as distinct. `test_set_construction.py` is the authoritative 1-gram implementation.

The released pair CSVs additionally report **1-, 2-, 3-, 5-, and 7-gram RPS**. `append_ngram_rps.py` defines the exact list as `NGRAMS = (1, 2, 3, 5, 7)` and computes the fraction of distinct target n-grams present in the reference.

## Included files

- `sample2pickle.py`: phonemizes LibriTTS-R reference transcripts with eSpeak and stores IPA phones plus 24-dimensional PanPhon articulatory features.
- `speech2pickle.py`: infers IPA phones from reference speech with a ZIPA ONNX checkpoint and stores the same articulatory representation.
- `test_set_construction.py`: creates length-controlled fixed-reference and fixed-target pairs while sampling a broad range of 1-gram RPS values.
- `append_ngram_rps.py`: adds the 1-, 2-, 3-, 5-, and 7-gram RPS columns to both pair CSVs.
- `data_indices/*.csv`: released TTS input-pair indices. `ref_id` and `tgt_id` are split-local parquet row indices.
- `data_indices/test.clean.pkl` and `data_indices/train.clean.100.pkl`: transcript-derived intermediate phone/feature dictionaries keyed by split-local row index.

## Setup

The original local Conda environment was named `phonemizer`; in this repository it is called `rps`:

```bash
conda create -n rps python=3.11 -y
conda activate rps
pip install -r requirements.txt
```

Install the eSpeak NG system executable required by `phonemizer` (for example, `sudo apt install espeak-ng` on Ubuntu). GPU inference in `speech2pickle.py` additionally requires a CUDA stack compatible with PyTorch and ONNX Runtime. If no CUDA provider is available, the script falls back to CPU; replace `onnxruntime-gpu` with `onnxruntime` for a CPU-only installation.

`requirements.txt` pins packages observed in the original environment. Audio/ONNX imports that were absent from that environment export are listed without invented version pins and should be pinned once the final paper environment is archived.

## Expected data layout

Download LibriTTS-R separately and arrange its parquet shards as follows (the corpus itself is not redistributed):

```text
database/libritts_r/
├── test.clean/test.clean-*.parquet
└── train.clean.100/train.clean.100-*.parquet
```

### Install the ZIPA checkpoint

`speech2pickle.py` uses the FP16 CTC checkpoint and token vocabulary from `anyspeech/zipa-large-crctc-300k`. Download only the required files into `pretrained/zipa/`:

```bash
huggingface-cli download anyspeech/zipa-large-crctc-300k \
  model.fp16.onnx tokens.txt \
  --local-dir pretrained/zipa
```

The resulting layout must be:

```text
pretrained/zipa/
├── model.fp16.onnx
└── tokens.txt
```

The checkpoint is downloaded separately and is not committed to this repository.

## Build the intermediate representations

Transcript-derived IPA and articulatory features:

```bash
python sample2pickle.py \
  --dataset-dir database/libritts_r \
  --output-dir data_indices
```

This writes `data_indices/test.clean.pkl` and `data_indices/train.clean.100.pkl`. Use `--limit N` for a smoke test and adjust `--jobs` to the available CPUs.

Speech-derived IPA and articulatory features for `test.clean`:

```bash
python speech2pickle.py \
  --dataset-dir database/libritts_r \
  --model-dir pretrained/zipa \
  --output data_indices/test.clean.s2p.pkl
```

The pickle schema is:

```python
{
    "<split-local row index>": {
        "spkid": "<speaker id>",
        "phonemes": ["<IPA phone>", ...],
        "articulatory": numpy.ndarray,  # [number_of_phones, 24]
    }
}
```

Pickle files must only be loaded from trusted sources.

## Construct the TTS input pairs

The construction script expects transcript-phone metadata for `test.clean` and `train.clean.100`. Each CSV must contain a phone column (`phonemes`, `phoneme`, `phonmes`, or `phones`) and may contain an utterance ID column. The reference metadata should also contain a speaker ID; if omitted, it is joined from the `test.clean` parquet row order.

```bash
python test_set_construction.py \
  --ref-metadata data_indices/test-clean.csv \
  --tgt-metadata data_indices/train-clean-100.csv \
  --dataset-dir database/libritts_r \
  --output-dir data_indices \
  --seed 42

python append_ngram_rps.py --data-dir data_indices
```

Outputs:

- `test.clean_x_train.clean.100_fixed_ref.csv`: each fixed reference is paired with multiple target texts.
- `test.clean_x_train.clean.100_fixed_tgt.csv`: each fixed target is paired with multiple references from a speaker.

Within each group, the script controls phoneme-length variation and selects pairs across RPS bins. The second command appends `1-gram`, `2-gram`, `3-gram`, `5-gram`, and `7-gram`; therefore 7-gram RPS is explicitly included.

## Generate and evaluate TTS audio

The local research workspace used the following independently maintained TTS implementations: [Amphion](https://github.com/open-mmlab/Amphion), [CosyVoice](https://github.com/QwenAudio/CosyVoice), [F5-TTS](https://github.com/SWivid/F5-TTS), [XTTS wrapper](https://github.com/Render-AI-Team/cog-xtts), and [YourTTS](https://github.com/edresson/yourtts). Download the desired implementation separately and follow its own `README.md` for installation, checkpoints, reference-conditioned inference, and evaluation. These projects have different and sometimes conflicting dependencies, so they are intentionally not vendored into this repository or merged into the `rps` environment.

For every pair CSV row:

1. Resolve `ref_id` to the corresponding `test.clean` reference waveform.
2. Resolve `tgt_id` to the corresponding `train.clean.100` target transcript.
3. Synthesize the target text using the reference waveform as the speaker prompt.
4. Save one waveform per pair, for example `ref_<ref_id>__tgt_<tgt_id>.wav`.
5. Preserve `ref_id`, `tgt_id`, all RPS columns, reference path, generated path, model/checkpoint, seed, and decoding settings in a manifest.

Use the same checkpoint and decoding configuration for all RPS conditions. Do not use the target utterance's original audio as a speaker prompt.

## Speaker similarity conditioned on RPS

Use a fixed pretrained speaker-embedding model for both the generated and reference waveforms. Apply the model's required sample rate, normalization, and segmentation identically, then compute cosine similarity:

$$
\operatorname{sim}(r,\hat{x})=
\frac{e_r^\top e_{\hat{x}}}
{\lVert e_r\rVert_2\lVert e_{\hat{x}}\rVert_2}.
$$

Write one result row per synthesized pair and retain its RPS columns. Report the overall mean with uncertainty and summarize similarity by predeclared RPS bins. The two controlled sets answer complementary questions:

- **Fixed reference:** how similarity changes with target-phone support while reference identity/audio is held fixed.
- **Fixed target:** how similarity changes across same-speaker references while target text is held fixed.

Aggregate with the grouping structure intact (for example, group-aware bootstrap confidence intervals or a mixed-effects model), rather than treating all rows as independent. Record the speaker model name, checkpoint, embedding layer, sample rate, and scoring protocol.

## Reproducibility notes

- Pair sampling is deterministic for a fixed seed.
- The main `rps` column is the 1-gram set overlap; the released diagnostic columns extend it through 7-gram.
- IDs are split-local row indices and depend on sorted parquet shard order.
- This repository does not redistribute LibriTTS-R audio or the ZIPA checkpoint.
- See `github_upload.md` for the remaining public-release checklist.
