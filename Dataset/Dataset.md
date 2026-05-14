# 📦 BirdCLEF+ 2026: Comprehensive Dataset Documentation
> **Generated from**: Kaggle Dataset Profiling, Competition Overview, Public Leaderboard Analysis  
> **Last Updated**: May 14, 2026 | **Total Size**: 129.11 GB | **Total Files**: 92,477  
> **Target Metric**: Macro-averaged ROC-AUC | **Constraint**: CPU-only, ≤90 min runtime

---

## 📑 Table of Contents
1. [Dataset Overview & Hierarchy](#1-dataset-overview--hierarchy)
2. [ Core Competition Data (`competitions/`)](#2-core-competition-data-competitions)
3. [🤖 Pre-trained Models & Runtime Assets (`datasets/` & `models/`)](#3-pre-trained-models--runtime-assets-datasets--models)
4. [💾 Pre-computed Features & Inference Caches (`datasets/` & `notebooks/`)](#4-pre-computed-features--inference-caches-datasets--notebooks)
5. [🎵 Audio Data Specifications](#5-audio-data-specifications)
6. [🏷️ Label Schema & Taxonomy Mapping](#6-label-schema--taxonomy-mapping)
7. [📊 Statistical Profile & Distribution](#7-statistical-profile--distribution)
8. [ Pipeline Integration Guide](#8-pipeline-integration-guide)
9. [⚠️ Important Notes & Profiling Caveats](#9-important-notes--profiling-caveats)
10. [📎 References & External Links](#10-references--external-links)

---

## 1. Dataset Overview & Hierarchy

The BirdCLEF+ 2026 dataset is structured into **4 primary directories** on Kaggle, optimized for CPU-only inference, transfer learning, and fast experimentation:

```
/kaggle/input/
├── competitions/          (~10.2 GB)  → Core competition data (audio, labels, taxonomy)
├── datasets/              (~115.0 GB) → Pre-trained ONNX models, wheels, cached features
├── models/                (~391 MB)   → Google Perch v2 metadata & label mappings
└── notebooks/             (~623 MB)   → TF/TensorBoard wheels, OOF features, submissions
```

**Key Characteristics:**
- **Audio Format**: `.ogg` (Opus), 32kHz, mono, variable bitrates
- **Label Format**: Multi-hot, weak supervision (clip-level) + strong supervision (window-level soundscapes)
- **Species Count**: 234 classes across 5 taxa (Aves, Amphibia, Mammalia, Reptilia, Insecta)
- **Test Set**: ~600 hidden 1-minute soundscapes, split into 12×5s windows per file
- **Runtime Constraint**: GPU disabled during submission; all models must be ONNX/TorchScript

---

## 2.  Core Competition Data (`competitions/`)

### 2.1 Metadata Files
| File | Description | Key Columns |
|------|-------------|-------------|
| `train.csv` | Weakly labeled recordings from Xeno-canto & iNaturalist | `primary_label`, `secondary_labels`, `latitude`, `longitude`, `rating`, `collection`, `filename` |
| `taxonomy.csv` | Species metadata & class mapping | `primary_label`, `scientific_name`, `class_name` (Aves/Amphibia/etc.), `iNat_taxon_id` |
| `train_soundscapes_labels.csv` | Strong window-level annotations for Pantanal field recordings | `filename`, `start`, `end`, `primary_label` (semicolon-separated) |
| `sample_submission.csv` | Submission template | `row_id`, 234 species probability columns |
| `recording_location.txt` | High-level site metadata | Location, habitat type, recorder model |

### 2.2 Audio Structure
- **Training Clips**: `train_audio/` → Short recordings (typically 5–60s), weakly labeled, globally distributed
- **Training Soundscapes**: `train_soundscapes/` → Continuous 1-min recordings from Pantanal sites, partially labeled
- **Test Soundscapes**: `test_soundscapes/` (hidden) → ~600 files, fully unlabeled, used for scoring

> 💡 **Critical Insight**: Some species in the hidden test set **only appear in `train_soundscapes_labels.csv`**, not in `train_audio`. Exploiting soundscape labels is mandatory for top performance.

---

## 3. 🤖 Pre-trained Models & Runtime Assets (`datasets/` & `models/`)

### 3.1 ONNX Models (CPU-Optimized)
| Model | Path | Purpose | Input/Output |
|-------|------|---------|--------------|
| `perch_v2_no_dft.onnx` | `datasets/` | Google Perch v2 (no-DFT variant) | `(N, 160000)` → `logits(14795)`, `embedding(1536)` |
| `perch_v2.onnx` | `datasets/` | Standard Perch v2 (fallback) | Same as above |
| `sed_fold0.onnx` – `sed_fold4.onnx` | `datasets/` | 5-fold EfficientNet-B2 SED ensemble | `(B, 1, 128, T)` → `clip_logits`, `frame_logits` |

### 3.2 Python Wheels & Runtimes
| Package | Purpose |
|---------|---------|
| `onnxruntime-1.24.4-cp312...whl` | CPU inference engine (4-thread optimized) |
| `openvino-2026.0.0...whl` | Alternative inference backend (optional) |
| `openvino_telemetry-2025.2.0...whl` | OpenVINO telemetry |
| `tensorflow-2.20.0...whl` | TF fallback for Perch/SED compatibility |
| `tensorboard-2.20.0...whl` | Logging & debugging |

### 3.3 Label Mappings (`models/`)
- `labels.csv` → Perch's 14,795-class vocabulary
- `perch_v2_ebird_classes.csv` → eBird code mapping for bird species
- Used to align Perch outputs with competition's 234 classes via `scientific_name`

---

## 4. 💾 Pre-computed Features & Inference Caches (`datasets/` & `notebooks/`)

### 4.1 Perch Embeddings Cache
| File | Format | Shape/Size | Content |
|------|--------|------------|---------|
| `full_perch_arrays.npz` | NumPy | `(N, 1536)` + `(N, 234)` | Perch embeddings & mapped logits for all labeled windows |
| `full_perch_meta.parquet` | Parquet | `row_id`, `filename`, `site`, `hour_utc` | Metadata aligned with embeddings |

### 4.2 OOF & Pseudo-label Features
| File | Format | Purpose |
|------|--------|---------|
| `full_oof_meta_features.npz` | NumPy | Out-of-fold predictions, confidence scores, calibration metadata |
| `pseudo_window_meta.csv` | CSV | Pseudo-labeled window indices, confidence thresholds |
| `soundscape_cache_meta.csv` | CSV | Cached SED/Perch features for train soundscapes |
| `audio_cache_meta.csv` | CSV | Mapping between raw audio and cached feature indices |

### 4.3 Notebook Artifacts (`notebooks/`)
- `full_oof_meta_features.npz` (duplicate/reference)
- `submission.csv` (baseline/template)
- TF & TensorBoard wheels for local debugging

> ⚡ **Performance Impact**: Using cached embeddings skips ~2–3 minutes of Perch inference per run. Critical for staying under 90-min CPU limit.

---

## 5. 🎵 Audio Data Specifications

| Property | Value |
|----------|-------|
| **Sample Rate** | 32,000 Hz (fixed) |
| **Channels** | Mono (stereo downmixed if present) |
| **Duration** | 60 seconds per soundscape; variable for clips |
| **Format** | OGG Opus, resampled from original sources |
| **Windowing** | 12 non-overlapping 5-second windows per file |
| **File Naming** | `BC2026_[Train/Test]_[ID]_[Site]_[YYYYMMDD]_[HHMMSS].ogg` |

**Example**: `BC2026_Test_0001_S05_20250227_010002.ogg`  
→ Site `S05`, Feb 27 2025, 01:00 UTC, 5-second window ends at `20` → `row_id: ..._0002_20`

---

## 6. 🏷️ Label Schema & Taxonomy Mapping

### 6.1 Class Distribution
| Taxon | Count | % of Data | Notes |
|-------|-------|-----------|-------|
| **Aves** | 162 | ~97.9% | eBird codes (e.g., `bmwool`, `breant1`) |
| **Amphibia** | 35 | ~1.3% | iNat IDs, few-shot challenge |
| **Insecta** | 28 | ~0.6% | Sonotypes (`47158son16`), acoustic clusters |
| **Mammalia** | 8 | ~0.3% | Jaguars, capybaras, bats |
| **Reptilia** | 1 | ~0.003% | Single species, extreme imbalance |

### 6.2 Label Types
- **Primary Label**: Target species for prediction
- **Secondary Labels**: Additional species noted by annotators (incomplete, noisy)
- **Multi-label Format**: Semicolon-separated in soundscapes (e.g., `bmwool;breant1`)
- **Sonotype Groups**: Insect clusters with visually similar spectrograms; require max-pooling during post-processing

### 6.3 Mapping Strategy
1. Match `primary_label` → `scientific_name` via `taxonomy.csv`
2. Map to Perch's 14,795 classes using `models/labels.csv`
3. Unmapped classes → use genus-level proxy or rely on SED pipeline

---

## 7.  Statistical Profile & Distribution

### 7.1 Overall Metrics
- **Total Size**: 129.11 GB
- **Total Files**: 92,477
- **Audio Files**: ~46,232 (`.ogg`)
- **Feature/Model Files**: ~46,245 (`.npz`, `.parquet`, `.onnx`, `.whl`, `.csv`)

### 7.2 Top 10 Species by Training Frequency
| Rank | eBird/iNat Code | Approx. Count |
|------|-----------------|---------------|
| 1 | `banana` | 498 |
| 2 | `coffal1` | 495 |
| 3 | `compau` | 493 |
| 4 | `bncfly` | 492 |
| 5 | `bobfly1` | 492 |
| 6 | `blewoc` | 485 |
| 7 | `bmwool` | 478 |
| 8 | `breant1` | 472 |
| 9 | `fepowl` | 410 |
| 10 | `compco1` | 385 |

> 🔍 **Note**: Profiling sampled ~84 classes; full competition uses 234. Long-tail distribution confirmed: ~30% of species have <10 training examples.

### 7.3 Temporal & Spatial Patterns
- **Sites**: `S01`–`S25+` across Pantanal wetlands
- **Hours**: 24-hour UTC coverage; dawn chorus (04:00–08:00 UTC) shows highest bird activity
- **Seasonality**: Recordings span 2021–2025; dry vs. wet season variations present

---

## 8. 🔗 Pipeline Integration Guide

### 8.1 Recommended Kaggle Setup
```python
# Attach these datasets to your notebook:
1. birdclef-2026 (competitions/)
2. tuckerarrants/perch-v2-no-dft-onnx (datasets/)
3. tuckerarrants/bc2026-distilled-sed-public (datasets/)
4. jaejohn/perch-meta (datasets/)
5. rishikeshjani/perch-onnx-for-birdclef-2026 (datasets/)
6. ashok205/tf-wheels (notebooks/)
```

### 8.2 Loading Pre-computed Features
```python
import numpy as np
import pandas as pd

# Load Perch cache
emb = np.load("/kaggle/input/datasets/full_perch_arrays.npz")["embs"]  # (N, 1536)
meta = pd.read_parquet("/kaggle/input/datasets/full_perch_meta.parquet")

# Align with labels
labels_df = pd.read_csv("/kaggle/input/competitions/train_soundscapes_labels.csv")
# ... merge on filename + window boundaries
```

### 8.3 ONNX Inference Template
```python
import onnxruntime as ort

so = ort.SessionOptions()
so.intra_op_num_threads = 4
session = ort.InferenceSession("/kaggle/input/datasets/perch_v2_no_dft.onnx", 
                               sess_options=so, providers=["CPUExecutionProvider"])

# Inference
outs = session.run(None, {"input": audio_batch})  # audio_batch: (B, 160000) float32
logits = outs[0].astype(np.float32)
embeddings = outs[1].astype(np.float32)
```

---

## 9. ⚠️ Important Notes & Profiling Caveats

1. **Hidden Test Set**: `test_soundscapes/` is not included in profiling. Structure matches `train_soundscapes/` but dates/times differ.
2. **Class Count Discrepancy**: Profiler detected 84 classes due to sampling; competition officially uses 234. Always use `sample_submission.csv` columns for alignment.
3. **File Type Pie Chart**: Shows 50% `.ogg` / 50% `.pt`. `.pt` refers to cached PyTorch tensors in feature directories; raw audio remains `.ogg`.
4. **CPU Runtime**: ONNX models are mandatory. TF/PyTorch models will exceed 90-min limit if not exported.
5. **License**: Data follows Xeno-canto & iNaturalist licensing; commercial use requires attribution & compliance with source terms.
6. **Versioning**: All assets are Version 1/2/3 as noted in dataset explorer. Do not mix mismatched versions.

---

## 10. 📎 References & External Links

| Resource | Link |
|----------|------|
| Competition Page | https://www.kaggle.com/competitions/birdclef-2026 |
| BirdCLEF+ 2026 Overview | `README.md` (provided) |
| Google Perch v2 Paper | https://arxiv.org/abs/2305.06245 |
| Xeno-canto API | https://xeno-canto.org/develop/api |
| iNaturalist Taxonomy | https://www.inaturalist.org/pages/developers |
| ONNX Runtime Docs | https://onnxruntime.ai/docs/ |
| Public LB 0.946 Notebook | https://www.kaggle.com/code/imaadmahmood/birdclef-2026-onnx-perch-sequence-modeling |

---

> 🦜 **Pro Tip**: Combine `full_perch_arrays.npz` with `train_soundscapes_labels.csv` to train your LightProtoSSM/MLP probes without re-running Perch. This saves ~150 seconds of CPU time and leaves budget for ensemble post-processing.

**Document Version**: 1.0 | **Author**: AI Data Engineering Specialist | **Date**: May 14, 2026  
*Use this documentation as your single source of truth for dataset navigation, pipeline design, and competition strategy.* 🚀
