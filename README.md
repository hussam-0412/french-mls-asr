# French ASR on Multilingual LibriSpeech

Project 3 compares automatic speech recognition systems on 2,426 French audiobook clips from the official Multilingual LibriSpeech (MLS) test split. The team trained its MFCC + GMM-HMM baseline on a one-hour sample of the official training split and ran the pretrained Whisper and wav2vec 2.0 baselines on the same test clips.

## Results on the official test split

| Baseline | Checkpoint | Revision or SHA-256 | WER | CER |
| --- | --- | --- | ---: | ---: |
| Whisper small | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | `973afd24965f72e36ca33b3055d56a652f456b4d` | 13.51% | 5.47% |
| wav2vec 2.0 + CTC | [jonatasgrosman/wav2vec2-large-xlsr-53-french](https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-french) | `7c79e105a6525d38e1e69f640b974b4a679723cc` | 15.24% | 4.41% |
| MFCC + GMM-HMM (team-trained) | [gmm-hmm.joblib](trained/gmm-hmm.joblib) | `32e594106c5fb4aa6a90e32b65915e5fa4e72cb5df448c836f16bfc7ab4eda0a` | 99.47% | 77.60% |

Source: [full_test_scores.csv](trained/full_test_scores.csv). The Whisper and wav2vec 2.0 checkpoints were used as published. The team implemented and trained the GMM-HMM baseline with 13-dimensional MFCC features, diagonal Gaussian mixtures per character, character bigram transitions, and up to four rounds of transcript-constrained Viterbi re-alignment. GMM fitting uses seed 42.

### Remaining baseline runs

French support is documented by the checkpoint cards. The runner has entries and revision pins for these models. Saved French MLS test predictions and scores are pending.

| Baseline | Checkpoint | Runner option |
| --- | --- | --- |
| CTC encoder | [nvidia/stt_fr_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc), CTC decoder | `ctc` |
| Attention encoder-decoder | [speechbrain/asr-crdnn-commonvoice-fr](https://huggingface.co/speechbrain/asr-crdnn-commonvoice-fr) | `attention` |
| RNN-T | [nvidia/stt_fr_fastconformer_hybrid_large_pc](https://huggingface.co/nvidia/stt_fr_fastconformer_hybrid_large_pc), RNN-T decoder | `rnnt` |
| HuBERT + CTC | [naver/mHuBERT-147-ASR-fr](https://huggingface.co/naver/mHuBERT-147-ASR-fr) | `hubert` |
| Canary | [nvidia/canary-1b-v2](https://huggingface.co/nvidia/canary-1b-v2) | `canary` |
| Parakeet | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | `parakeet` |
| Qwen3-ASR | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | `qwen` |

### Whisper fine-tuning

The benchmark has no test result for a team-fine-tuned Whisper checkpoint. The prediction script supports loading a local fine-tuned checkpoint. Reproducing this experiment requires the training script, the training settings, and the development checkpoint-selection record.

## Dataset and split audit

[Multilingual LibriSpeech, French (OpenSLR 94)](https://www.openslr.org/94/) is derived from LibriVox audiobooks. The original French release supplies the train, development, and test splits.

| Split | Speakers | Utterances | Hours |
| --- | ---: | ---: | ---: |
| Official train | 142 | 258,213 | 1,076.6 |
| Team training sample, selected from train | 36 | 239 | 1.0 |
| Official development | 18 | 2,416 | 10.1 |
| Official test | 18 | 2,426 | 10.1 |

The team's split audit reports zero shared speakers, recordings, or clip IDs between the official splits and zero verbatim test utterances in the official training split. These checks describe the dataset splits used by the project. Some published checkpoints list MLS among their training corpora, including Canary and Parakeet. The overlap audit does not cover their upstream training.

The train sample uses the IDs in `project3/train_1h_handles.txt`. It is a team-selected subset of the official training split, reported as sampled with seed 42. The handle list fixes the selection used by the trainer.

The test transcripts in `mls/test/transcripts.txt` come from the official release. The project package contains the test transcript metadata and predictions. Rerunning audio inference and the split audit requires the original French MLS audio and split metadata. MLS data are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Preprocessing and scoring

The audio loader averages channels to mono and resamples to 16 kHz when needed. It resolves clips by utterance ID and checks that training handles belong to the requested split. The scoring script applies the same processing to each reference and hypothesis: NFC Unicode normalization, lowercase, apostrophe normalization, punctuation replacement, and whitespace collapse. Accents, apostrophes, and digits are retained. CER counts spaces between the normalized words.

WER divides the total word substitutions, deletions, and insertions by the total reference word count. It measures the error at the level of words in the transcript. CER applies the same calculation at the character level and helps show partial spelling errors, including French accent mistakes. The scorer aggregates edits over the corpus and requires exact coverage of test IDs. [Hugging Face Audio Course, Chapter 5](https://huggingface.co/learn/audio-course/en/chapter5/evaluation).

## Reproduce the reported scores

From the repository root, use Python 3 and the included test transcripts and predictions.

~~~bash
python3 project3/evaluate.py mls \
  trained/whisper.json trained/wav2vec2.json trained/gmm-hmm.json \
  --split test --out rescored.csv
~~~

To rerun the split audits, supply a full French MLS directory with the release's `transcripts.txt` and `segments.txt` for train, dev, and test.

~~~bash
python3 project3/audit_mls_splits.py /path/to/mls_french
python3 project3/audit_mls_splits.py /path/to/mls_french \
  --train-handles project3/train_1h_handles.txt
~~~

For inference, extract the French MLS release with the original `<split>/audio/<speaker>/<book>/` layout. The current `project3/requirements.txt` lists the main dependencies. The NVIDIA runners also import NeMo, and the Qwen runner imports `qwen_asr`. The complete pinned inference environments remain to be recorded. Use development clips for smoke tests, then score test predictions using the command above.

~~~bash
python3 project3/predict.py /path/to/mls_french whisper \
  --split dev --limit 2 --out dev_whisper.json
python3 project3/predict.py /path/to/mls_french whisper \
  --split test --out whisper_test.json
~~~

The classical training and inference commands use the same stored training handles. Use a separate output location to preserve the reported run.

~~~bash
python3 project3/classical.py /path/to/mls_french \
  --handles project3/train_1h_handles.txt --out rerun/gmm-hmm.joblib
python3 project3/predict.py /path/to/mls_french gmm-hmm \
  --checkpoint rerun/gmm-hmm.joblib --split test --out rerun/gmm-hmm.json
~~~

## Repository layout

| Path | Purpose |
| --- | --- |
| `README.md` | Dataset, benchmark, and reproduction instructions |
| `project3/audit_mls_splits.py` | Split sizes, durations, and overlap checks |
| `project3/common.py` | Audio loading and transcript normalization |
| `project3/classical.py` | MFCC + GMM-HMM training and decoding |
| `project3/predict.py` | Per-model inference and prediction export |
| `project3/evaluate.py` | WER/CER scoring and evaluation CSV generation |
| `project3/model_revisions.json` | Published checkpoint revision pins |
| `project3/requirements.txt` | Main inference and training dependencies |
| `project3/train_1h_handles.txt` | Utterance IDs for the training sample |
| `mls/test/transcripts.txt` | Official French MLS test references |
| `trained/*.json` | Per-utterance predictions and model metadata |
| `trained/full_test_scores.csv` | Three-system full-test results generated by the evaluator |
| `trained/whisper_scores.csv` | Whisper-only export of the baseline scores |
| `trained/gmm-hmm.joblib` | Trained classical checkpoint used for the reported results |
| `contributions/` | One contribution record per team member, added by its author |
