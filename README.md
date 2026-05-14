# 🦜 BirdCLEF+ 2026 — Hướng Dẫn Toàn Diện Chiến Thắng
> **Pantanal Acoustic Species Identification | 234 Species | Macro ROC-AUC | CPU-only 90 min**
> 
> *Phiên bản dành cho người muốn top 5% — Tổng hợp từ: EDA Reports, Top 2025 Solutions, Public LB 0.946 Notebook*

---

## 📋 MỤC LỤC

1. [Tổng Quan Cuộc Thi](#1-tổng-quan-cuộc-thi)
2. [Khung Lý Thuyết Cốt Lõi](#2-khung-lý-thuyết-cốt-lõi)
3. [Phân Tích Dữ Liệu (EDA) — Những Điều Bắt Buộc Phải Biết](#3-phân-tích-dữ-liệu-eda)
4. [Kiến Trúc Pipeline Tổng Thể](#4-kiến-trúc-pipeline-tổng-thể)
5. [Phase 1 — Thiết Lập Môi Trường & Dữ Liệu](#5-phase-1--thiết-lập-môi-trường--dữ-liệu)
6. [Phase 2 — Feature Extraction với Perch/BirdNET](#6-phase-2--feature-extraction)
7. [Phase 3 — SED Model (EfficientNet Backbone)](#7-phase-3--sed-model)
8. [Phase 4 — Sequence Modeling (SSM/Transformer)](#8-phase-4--sequence-modeling)
9. [Phase 5 — Pseudo-Labeling & Semi-Supervised](#9-phase-5--pseudo-labeling--semi-supervised)
10. [Phase 6 — Post-Processing Stack](#10-phase-6--post-processing-stack)
11. [Phase 7 — Ensemble Strategy](#11-phase-7--ensemble-strategy)
12. [Chiến Lược CPU Inference Tối Ưu](#12-chiến-lược-cpu-inference-tối-ưu)
13. [Lộ Trình Thực Chiến Tuần-Theo-Tuần](#13-lộ-trình-thực-chiến-tuần-theo-tuần)
14. [Checklist Trước Khi Submit](#14-checklist-trước-khi-submit)
15. [Code Templates & Snippets](#15-code-templates--snippets)
16. [Tài Liệu Tham Khảo](#16-tài-liệu-tham-khảo)

---

## 1. TỔNG QUAN CUỘC THI

### 1.1 Bài Toán

```
Input:  Audio soundscape 1 phút (60s) → cắt thành 12 cửa sổ 5 giây
Output: P(species_i | window_j) cho 234 loài × 12 windows = 2,808 predictions/file
Metric: Macro-averaged ROC-AUC (bỏ qua class không có true positive trong test)
```

### 1.2 Ràng Buộc Cứng (Hard Constraints)

| Ràng Buộc | Chi Tiết | Tác Động |
|-----------|----------|----------|
| CPU-only | GPU bị disable (1 min runtime) | Phải ONNX/TorchScript tất cả |
| 90 phút CPU | ~600 test files × 1 min = 10 min audio | ~9 giây/file budget |
| Không internet | Mọi model phải pre-packaged | Kaggle Datasets |
| ~600 test soundscapes | 1 phút mỗi file, 32kHz | ~36,000 windows |

### 1.3 Phân Bố Loài (Biết để Chiến Đấu)

```
Aves (Chim):    162 loài — 97.9% training data  ← Domain quen
Amphibia (ếch): 35 loài  — 1.3% training data   ← FEW-SHOT CHALLENGE
Insecta:        28 loài  — 0.6% training data   ← Sonotypes đặc biệt
Mammalia:        8 loài  — 0.3% training data   ← Jaguar, Capybara...
Reptilia:        1 loài  — 0.003% training data ← 1 recording duy nhất!
```

**⚡ Key Insight:** 28 loài Insecta là "sonotypes" (không có species ID) — cần xử lý riêng!

---

## 2. KHUNG LÝ THUYẾT CỐT LÕI

### 2.1 Bài Toán Dưới Góc Độ ML

**Multi-label, multi-class, weak-supervision, domain-shift problem:**

```
[Weak Labels]          [Strong Labels]
XC/iNat recordings  →  train_soundscapes_labels.csv
(clip-level, noisy)    (window-level, expert-annotated, 59 files)
        ↓                        ↓
   35,549 clips            708 labeled windows
        ↓                        ↓
         [Semi-supervised Learning]
                  ↓
         [Domain Adaptation]
         XC/iNat → Pantanal field recordings
```

### 2.2 Tại Sao ROC-AUC Thay Vì Accuracy/F1?

ROC-AUC đo **khả năng xếp hạng** — không cần chọn threshold. Điều này có nghĩa:
- Calibration của probability quan trọng
- Rank ordering trong cùng file quan trọng hơn absolute value
- Macro = mỗi loài được weighted bằng nhau → loài hiếm cực kỳ quan trọng

### 2.3 Lý Thuyết Mel Spectrogram

```python
# Tín hiệu âm thanh → Feature cho CNN
Audio (32kHz, 5s = 160,000 samples)
    ↓  STFT (n_fft=2048, hop=512)
Power Spectrogram (1025 bins × 313 frames)
    ↓  Mel Filterbank (128 or 256 bins)
Mel Spectrogram (128 × 313)
    ↓  Log compression
Log-Mel Spectrogram → CNN input như "ảnh"
```

**Tại sao Mel scale?** Tai người (và nhiều loài) nhạy hơn với sự thay đổi tương đối tần số (logarithmic), không phải tuyệt đối. Mel filterbank mô phỏng điều này.

### 2.4 Lý Thuyết Transfer Learning Cho Bioacoustics

```
ImageNet Pretrain  →  AudioSet Pretrain  →  Bird/Wildlife Fine-tune
    (texture)           (sound events)        (species-specific)

Perch (Google):  AudioSet+bird → 14,795 species logits + 1536-dim embedding
BirdNET:         XC 6000+ species → strong bird-specific features
EfficientNet:    ImageNet → fine-tune trên mel spectrogram (treat as image)
```

**Perch embedding (1536-dim) là gold standard** cho bài toán này vì đã được train trên audio bird data khổng lồ.

### 2.5 Sound Event Detection (SED) vs. Clip-Level Classification

```
Clip-Level:  CNN nhìn toàn bộ 5s mel → predict probability
             + Đơn giản  + Robust với noise
             - Mất temporal detail

SED:         CNN với temporal pooling → frame-by-frame prediction
             → max-pool or attention-pool → clip prediction
             + Tốt hơn khi sound ngắn trong window
             + Cung cấp strong pseudo-labels
```

### 2.6 State Space Models (SSM/Mamba) cho Sequential Audio

```
Window 1 → Window 2 → ... → Window 12  (temporal sequence trong 1 file)

RNN/LSTM:  gradient vanishing, slow
Transformer: O(T²) attention, too slow for CPU
SSM (Mamba): O(T) recurrence + selective state, FAST trên CPU

LightProtoSSM trong notebook public:
  - Bidirectional SSM (forward + backward)
  - Cross-attention layers
  - Prototype-based classification
  - ~11 giây training trên 59 files!
```

---

## 3. PHÂN TÍCH DỮ LIỆU (EDA)

### 3.1 Những Con Số Quan Trọng Nhất

```python
# Phải chạy EDA này đầu tiên:

import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("/kaggle/input/birdclef-2026")
train = pd.read_csv(BASE / "train.csv")
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sc_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()  # 234 species
N_CLASSES = len(PRIMARY_LABELS)

# 1. Phân bố số lượng recording theo species
counts = train['primary_label'].value_counts()
print(f"Species với < 10 recordings: {(counts < 10).sum()}")
print(f"Species với < 5 recordings:  {(counts < 5).sum()}")

# 2. Số labeled windows trong train_soundscapes
print(f"\nLabeled soundscape windows: {len(sc_labels)}")
print(f"Unique files: {sc_labels['filename'].nunique()}")
print(f"Species xuất hiện trong soundscapes: {sc_labels['primary_label'].str.split(';').explode().nunique()}")

# 3. Class imbalance trong soundscape labels
all_labels = sc_labels['primary_label'].str.split(';').explode()
label_counts = all_labels.value_counts()
print(f"\nTop 10 species trong soundscapes:\n{label_counts.head(10)}")
print(f"\nSpecies chỉ xuất hiện trong soundscapes (không có XC): ???")
```

### 3.2 Domain Gap Analysis

```
XC/iNat recordings:              Pantanal test soundscapes:
- Studio/field, clean            - Continuous ambient recording
- Single species focus           - Multi-species overlap
- Submitter-selected moments     - Random time windows
- Global geographic diversity    - Pantanal-specific habitat
- Variable equipment             - Standardized recorders (32kHz OGG)

→ Khoảng cách domain LỚN nhất trong BirdCLEF history!
→ Giải pháp: train_soundscapes data cực kỳ quan trọng
```

### 3.3 Critical Data Notes

```
⚠️ QUAN TRỌNG: "Some species with occurrences in the hidden test data 
might only have train samples in the labeled portion of train_soundscapes 
and NOT in train_audio"

→ train_soundscapes_labels.csv là VÀNG — phải khai thác tối đa!
→ 59 fully-labeled files × 12 windows = 708 training examples for SSM
→ Chỉ 71 classes có ít nhất 1 positive trong 708 windows này
```

### 3.4 Sonotype Analysis (Insecta)

```python
# Sonotypes là insects với format: "47158son16"
# Tất cả cùng iNaturalist taxon 47158 (Gryllus?)
# Được phân biệt bằng âm thanh, không phải taxonomy

SONOTYPE_GROUPS = [
    ("47158son15", "47158son16"),   # visually similar spectrograms
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23")
]
# → Cần "sonotype mirroring": max-pool predictions trong cùng nhóm
```

---

## 4. KIẾN TRÚC PIPELINE TỔNG THỂ

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE (offline)                         │
│                                                                     │
│  train_audio (35k XC/iNat) ──┐                                      │
│                               ├─→ [SED Model Training]              │
│  train_soundscapes ──────────┘     EfficientNet-B2/B4               │
│     + pseudo-labels                5-fold CV                        │
│     + soundscape labels            ↓ ONNX export                    │
│                                                                     │
│  train_soundscapes ──────────→ [Perch Inference]                    │
│     (59 labeled files)            → embeddings (1536-dim)           │
│                                   → scores (234-dim)                │
│                                   ↓                                 │
│                              [LightProtoSSM Training]               │
│                                 40 epochs, ~11s                     │
│                              [MLP Probes Training]                  │
│                                 58 active classes                   │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ Checkpoint/Cache
┌─────────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE (submission, ≤90 min)            │
│                                                                     │
│  test_soundscapes (~600 files × 1min)                               │
│       ↓                                                             │
│  [ONNX Perch v2 no-DFT]  ←──── ~150x faster than TF SavedModel     │
│       ↓                                                             │
│  embeddings (1536) + logits (14,795 species)                        │
│       ↓                      ↓                                      │
│  [LightProtoSSM]      [MLP Probes]    [SED EfficientNet]            │
│  (train on-the-fly    (vectorized     (5-fold ONNX)                 │
│   40 epochs ~11s)      inference)                                   │
│       ↓                    ↓               ↓                        │
│  Proto scores        Prior-adjusted   SED probabilities             │
│       ↓                    ↓               ↓                        │
│       └──── ENSEMBLE (60% ProtoSSM / 40% SED) ────┘                │
│                              ↓                                      │
│              [5-Gate Post-Processing Stack]                         │
│  1. Noise suppression  2. Temporal continuity                       │
│  3. SED spike preservation  4. Sonotype mirroring                   │
│  5. Rare-class adaptive thresholding                                │
│                              ↓                                      │
│                      submission.csv                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. PHASE 1 — THIẾT LẬP MÔI TRƯỜNG & DỮ LIỆU

### 5.1 Kaggle Datasets Cần Attach

```python
# Datasets cần trong notebook (attach vào Kaggle notebook):

REQUIRED_DATASETS = {
    # Core competition
    "birdclef-2026": "Competition data",
    
    # Perch/Google model
    "google/bird-vocalization-classifier": "Perch v2 TF SavedModel + labels.csv",
    "rishikeshjani/perch-onnx-for-birdclef-2026": "ONNX Runtime wheel",
    "tuckerarrants/perch-v2-no-dft-onnx": "ONNX Perch (150x faster!)",
    
    # SED models
    "tuckerarrants/bc2026-distilled-sed-public": "5-fold EfficientNet SED ONNX",
    
    # Cache
    "jaejohn/perch-meta": "Pre-computed Perch embeddings for train_soundscapes",
    
    # TF wheels
    "ashok205/tf-wheels": "TF 2.20 for ONNX fallback",
}

# Optional nhưng mạnh:
OPTIONAL_DATASETS = {
    "birdclef-2025": "Previous year data for pseudo-labeling",
    "birdclef-2024": "More training data",
    "esc-50": "Background noise augmentation",
}
```

### 5.2 Cấu Trúc File Setup

```python
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa

# Paths
BASE = Path("/kaggle/input/birdclef-2026")
MODEL_DIR = Path("/kaggle/input/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")
WORK_DIR = Path("/kaggle/working/cache")
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Constants
SR = 32_000        # Sample rate
WINDOW_SEC = 5     # 5-second windows
WINDOW_SAMPLES = SR * WINDOW_SEC  # 160,000 samples
FILE_SAMPLES = 60 * SR            # 1,920,000 samples  
N_WINDOWS = 12                    # 60s / 5s = 12 windows

# Load metadata
taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

PRIMARY_LABELS = sample_sub.columns[1:].tolist()
N_CLASSES = len(PRIMARY_LABELS)  # 234
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}

print(f"✅ Setup complete: {N_CLASSES} classes, SR={SR}Hz")
```

### 5.3 Audio Loading & Preprocessing

```python
def read_60s(path: str) -> np.ndarray:
    """Load và normalize 60-second audio file."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    
    # Stereo → Mono
    if y.ndim == 2:
        y = y.mean(axis=1)
    
    # Resample nếu cần
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    
    # Pad hoặc truncate thành đúng 60 giây
    if len(y) < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - len(y)))
    else:
        y = y[:FILE_SAMPLES]
    
    return y  # shape: (1,920,000,)


def audio_to_windows(y: np.ndarray) -> np.ndarray:
    """Cắt 60s audio thành 12 windows 5 giây."""
    return y.reshape(N_WINDOWS, WINDOW_SAMPLES)  # (12, 160,000)


def compute_mel_spectrogram(
    y: np.ndarray,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmin: int = 20,
    fmax: int = 16000,
    top_db: int = 80
) -> np.ndarray:
    """Audio waveform → Log-Mel Spectrogram."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0
    )
    log_mel = librosa.power_to_db(mel, top_db=top_db)
    
    # Normalize per-clip
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    
    return log_mel.astype(np.float32)  # (n_mels, time_frames)
```

---

## 6. PHASE 2 — FEATURE EXTRACTION VỚI PERCH

### 6.1 Tại Sao Dùng Perch?

```
Google Perch v2:
- Trained on 14,795 bird species từ nhiều dataset lớn
- Output: 1536-dim embedding + 14,795-dim logit
- Đã được proven qua nhiều năm BirdCLEF
- ONNX version: 150x faster than TF SavedModel!

no-DFT variant:
- Thay thế DFT bằng learned filterbank
- Slightly khác nhau về outputs nhưng tương đương
- Prefer perch_v2_no_dft.onnx nếu có
```

### 6.2 ONNX Perch Setup

```python
import onnxruntime as ort
from pathlib import Path

def setup_perch_onnx(input_root: Path) -> tuple:
    """Setup ONNX Perch session."""
    # Tìm ONNX model
    onnx_path = next(
        (p for p in [
            next(input_root.glob("**/perch_v2_no_dft*.onnx"), None),
            next(input_root.glob("**/perch_v2*.onnx"), None),
        ] if p is not None),
        None
    )
    
    if onnx_path is None:
        raise FileNotFoundError("Không tìm thấy Perch ONNX!")
    
    # Tạo session với 4 threads
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=so,
        providers=["CPUExecutionProvider"]
    )
    
    input_name = session.get_inputs()[0].name
    out_map = {o.name: i for i, o in enumerate(session.get_outputs())}
    
    print(f"✅ ONNX Perch loaded: {onnx_path.name}")
    print(f"   Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"   Outputs: {list(out_map.keys())}")
    
    return session, input_name, out_map


def perch_inference(
    windows: np.ndarray,  # (N, 160000) float32
    session,
    input_name: str,
    out_map: dict
) -> tuple:
    """Chạy Perch inference trên batch windows."""
    outputs = session.run(None, {input_name: windows})
    logits = outputs[out_map["label"]].astype(np.float32)      # (N, 14795)
    embeddings = outputs[out_map["embedding"]].astype(np.float32)  # (N, 1536)
    return logits, embeddings
```

### 6.3 Species Mapping (Perch 14,795 → Competition 234)

```python
def build_species_mapping(taxonomy_df, bc_labels_df, primary_labels):
    """Map competition species sang Perch logit indices."""
    NO_LABEL = len(bc_labels_df)
    
    # Merge taxonomy với Perch labels theo scientific_name
    mapping = taxonomy_df.merge(
        bc_labels_df.rename(columns={"inat2024_fsd50k": "scientific_name"}),
        on="scientific_name",
        how="left"
    )
    mapping["bc_index"] = mapping["bc_index"].fillna(NO_LABEL).astype(int)
    
    lbl2bc = mapping.set_index("primary_label")["bc_index"]
    
    BC_INDICES = np.array(
        [int(lbl2bc.loc[c]) if c in lbl2bc.index else NO_LABEL 
         for c in primary_labels],
        dtype=np.int32
    )
    
    MAPPED_MASK = BC_INDICES != NO_LABEL
    MAPPED_POS = np.where(MAPPED_MASK)[0].astype(np.int32)
    MAPPED_BC_IDX = BC_INDICES[MAPPED_MASK].astype(np.int32)
    
    print(f"✅ Mapped: {MAPPED_MASK.sum()} / {len(primary_labels)} species")
    print(f"   Unmapped: {(~MAPPED_MASK).sum()} (cần proxy hoặc SED)")
    
    return BC_INDICES, MAPPED_MASK, MAPPED_POS, MAPPED_BC_IDX
```

### 6.4 Caching Strategy (Tiết Kiệm 2+ Phút)

```python
def build_or_load_perch_cache(
    soundscape_files: list,
    cache_dir: Path,
    perch_session,
    input_name: str,
    out_map: dict,
    force_rebuild: bool = False
):
    """Cache Perch embeddings để tránh recompute."""
    cache_meta = cache_dir / "perch_meta.parquet"
    cache_npz = cache_dir / "perch_arrays.npz"
    
    if not force_rebuild and cache_meta.exists() and cache_npz.exists():
        print("📂 Loading cached Perch features...")
        meta = pd.read_parquet(cache_meta)
        arrays = np.load(cache_npz)
        scores = arrays["scores"].astype(np.float32)
        embs = arrays["embs"].astype(np.float32)
        print(f"✅ Cache loaded: {scores.shape}")
        return meta, scores, embs
    
    print(f"🔄 Building Perch cache for {len(soundscape_files)} files...")
    # ... run inference và save
    # (xem full implementation trong notebook public)
```

---

## 7. PHASE 3 — SED MODEL

### 7.1 Architecture Overview

```python
import torch
import torch.nn as nn
import timm

class BirdSEDModel(nn.Module):
    """
    Sound Event Detection model cho BirdCLEF.
    Input: Mel Spectrogram (B, 1, n_mels, T)
    Output: clip_logits (B, 234) + frame_logits (B, T', 234)
    """
    def __init__(
        self,
        backbone: str = "tf_efficientnet_b2_ns",
        n_classes: int = 234,
        n_mels: int = 128,
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        
        # Backbone (treat mel as "image")
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            in_chans=1,           # Mono spectrogram
            drop_path_rate=drop_path_rate,  # StochasticDepth regularization
        )
        
        # Get feature dimension
        n_features = self.backbone.num_features
        
        # Remove original classifier
        self.backbone.reset_classifier(0)
        
        # SED head: frame-level predictions
        # Attention pooling thay vì simple max/avg
        self.att_block = AttentionPooling(n_features, n_classes)
        
        # Clip-level head
        self.clip_head = nn.Linear(n_features, n_classes)
        
    def forward(self, x):
        # x: (B, 1, n_mels, T) 
        # Extract features: (B, C, H', W')
        features = self.backbone.forward_features(x)
        
        # Global average pooling over frequency
        # features: (B, C, H', T') → (B, C, T') → (B, T', C)
        feat_temporal = features.mean(dim=2).permute(0, 2, 1)
        
        # Frame-level predictions
        frame_logits = self.att_block(feat_temporal)  # (B, T', n_classes)
        
        # Clip-level (max pooling over time)
        clip_feat = features.mean(dim=[2, 3])  # (B, C)
        clip_logits = self.clip_head(clip_feat)  # (B, n_classes)
        
        return clip_logits, frame_logits


class AttentionPooling(nn.Module):
    """Attention-based temporal pooling cho SED."""
    def __init__(self, in_features: int, n_classes: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Tanh(),
            nn.Linear(512, n_classes),
            nn.Softmax(dim=1)
        )
        self.classifier = nn.Linear(in_features, n_classes)
    
    def forward(self, x):
        # x: (B, T, C)
        attn = self.attention(x)        # (B, T, n_classes)
        pred = self.classifier(x)       # (B, T, n_classes)
        frame_out = pred * attn         # weighted predictions
        return frame_out
```

### 7.2 Augmentation Stack (Theo 2025 Winners)

```python
import numpy as np
import torch
import random

class AudioAugmentation:
    """Augmentation pipeline cho BirdCLEF."""
    
    def __init__(self, sr: int = 32000):
        self.sr = sr
    
    def mixup(
        self, 
        audio1: np.ndarray, 
        audio2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray,
        alpha: float = 0.5
    ):
        """
        Audio-domain MixUp (2025 2nd place: +3.6% Private LB!)
        Thực hiện trên waveform, KHÔNG phải spectrogram.
        """
        lam = np.random.beta(alpha, alpha)
        mixed_audio = lam * audio1 + (1 - lam) * audio2
        # Element-wise max cho multi-label
        mixed_label = np.maximum(label1, label2)
        return mixed_audio, mixed_label
    
    def random_filtering(self, audio: np.ndarray) -> np.ndarray:
        """
        Simulate microphone/channel variation.
        2025 2nd place: +~1% Private LB.
        """
        from scipy import signal
        # Random biquad filter
        freq = random.uniform(500, 8000)
        q = random.uniform(0.5, 5.0)
        gain_db = random.uniform(-6, 6)
        b, a = signal.iirpeak(freq / (self.sr / 2), q)
        return signal.lfilter(b, a, audio).astype(np.float32)
    
    def add_background_noise(
        self, 
        audio: np.ndarray, 
        noise: np.ndarray,
        snr_db: float = None
    ) -> np.ndarray:
        """Mix background noise (ESC-50 + soundscapes)."""
        if snr_db is None:
            snr_db = random.uniform(5, 20)
        
        # Đảm bảo cùng độ dài
        if len(noise) > len(audio):
            start = random.randint(0, len(noise) - len(audio))
            noise = noise[start:start + len(audio)]
        else:
            noise = np.pad(noise, (0, len(audio) - len(noise)))
        
        # Scale noise theo SNR
        signal_power = np.mean(audio ** 2) + 1e-9
        noise_power = np.mean(noise ** 2) + 1e-9
        scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
        
        return (audio + scale * noise).astype(np.float32)
    
    def time_shift(self, audio: np.ndarray, max_shift: float = 0.1) -> np.ndarray:
        """Random circular shift."""
        shift = int(random.uniform(-max_shift, max_shift) * len(audio))
        return np.roll(audio, shift)
    
    def spec_augment(
        self, 
        mel: np.ndarray,
        freq_mask_param: int = 20,
        time_mask_param: int = 30,
        n_freq_masks: int = 2,
        n_time_masks: int = 2
    ) -> np.ndarray:
        """SpecAugment: frequency + time masking."""
        mel = mel.copy()
        n_mels, n_frames = mel.shape
        
        for _ in range(n_freq_masks):
            f = random.randint(0, freq_mask_param)
            f0 = random.randint(0, n_mels - f)
            mel[f0:f0+f, :] = 0
        
        for _ in range(n_time_masks):
            t = random.randint(0, time_mask_param)
            t0 = random.randint(0, n_frames - t)
            mel[:, t0:t0+t] = 0
        
        return mel
```

### 7.3 Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftAUCLoss(nn.Module):
    """
    Direct AUC optimization (2025 1st place key innovation).
    Tốt hơn BCE cho ROC-AUC metric.
    """
    def __init__(self, margin: float = 0.2, gamma: float = 0.5):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred: (B, C), y_true: (B, C) ∈ [0, 1]
        losses = []
        for c in range(y_pred.shape[1]):
            pos_mask = y_true[:, c] > 0.5
            neg_mask = ~pos_mask
            
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            
            pos_preds = y_pred[pos_mask, c]  # (n_pos,)
            neg_preds = y_pred[neg_mask, c]  # (n_neg,)
            
            # Pairwise differences
            diff = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)  # (n_pos, n_neg)
            loss = torch.clamp(self.margin - diff, min=0) ** 2
            losses.append(loss.mean())
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)


class FocalLoss(nn.Module):
    """Focal Loss cho class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (self.alpha * focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """
    BCE + SoftAUC + Focal - kết hợp cho best results.
    Tỷ lệ điều chỉnh theo epoch (BCE nhiều hơn khi warm-up).
    """
    def __init__(self, bce_weight=0.5, auc_weight=0.3, focal_weight=0.2):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.auc = SoftAUCLoss()
        self.focal = FocalLoss()
        self.bce_w = bce_weight
        self.auc_w = auc_weight
        self.focal_w = focal_weight
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        return (
            self.bce_w * self.bce(logits, targets) +
            self.auc_w * self.auc(probs, targets) +
            self.focal_w * self.focal(logits, targets)
        )
```

### 7.4 Training Loop

```python
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_sed_model(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    device: str = "cuda",
    use_swa: bool = True,
    pos_weight_cap: float = 20.0,
) -> nn.Module:
    """Training loop với SWA và early stopping."""
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        epochs=n_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos"
    )
    
    criterion = CombinedLoss()
    
    # SWA setup
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=lr * 0.3)
        swa_start_epoch = int(n_epochs * 0.75)
    
    best_val_auc = 0.0
    best_state = None
    patience = 10
    wait = 0
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
            audio = batch["audio"].to(device)
            labels = batch["labels"].to(device)
            
            # Mixup (50% probability)
            if np.random.random() < 0.5:
                idx = torch.randperm(audio.size(0))
                lam = np.random.beta(0.5, 0.5)
                audio = lam * audio + (1 - lam) * audio[idx]
                labels = torch.maximum(labels, labels[idx])
            
            # Forward
            clip_logits, frame_logits = model(audio)
            
            # Loss: clip + frame
            clip_loss = criterion(clip_logits, labels)
            frame_max_logits = frame_logits.max(dim=1).values
            frame_loss = criterion(frame_max_logits, labels)
            loss = 0.7 * clip_loss + 0.3 * frame_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if epoch < swa_start_epoch:
                scheduler.step()
            
            train_loss += loss.item()
        
        # SWA update
        if use_swa and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Validation
        val_auc = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}: loss={train_loss/len(train_loader):.4f}, val_auc={val_auc:.4f}")
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best state
    model.load_state_dict(best_state)
    
    if use_swa:
        # Update BatchNorm statistics for SWA model
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        return swa_model
    
    return model
```

### 7.5 Export Sang ONNX (Bắt Buộc Cho CPU Inference)

```python
import torch
import onnx

def export_to_onnx(
    model: torch.nn.Module,
    output_path: str,
    n_mels: int = 128,
    n_frames: int = 313,  # ~5 giây @ hop=512, sr=32000
    opset: int = 14
):
    """Export PyTorch SED model sang ONNX."""
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 1, n_mels, n_frames)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["clip_logits", "frame_logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "clip_logits": {0: "batch_size"},
            "frame_logits": {0: "batch_size"},
        }
    )
    
    # Verify
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print(f"✅ ONNX model exported to {output_path}")
    
    # Benchmark
    import onnxruntime as ort
    import time
    
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    dummy_np = np.random.randn(12, 1, n_mels, n_frames).astype(np.float32)
    
    # Warm up
    session.run(None, {"input": dummy_np})
    
    t0 = time.time()
    for _ in range(10):
        session.run(None, {"input": dummy_np})
    print(f"   Inference speed: {(time.time()-t0)/10*1000:.1f}ms per batch of 12 windows")
```

---

## 8. PHASE 4 — SEQUENCE MODELING

### 8.1 Tại Sao Cần Temporal Modeling?

```
Bài toán: 12 windows/file KHÔNG độc lập!

Window 1  Window 2  ...  Window 12
 ────────────────────────────────
 Cùng location, cùng thời điểm
 Species không bật tắt ngẫu nhiên
 → Temporal continuity là prior knowledge mạnh!

Ví dụ:
- Một con chim hót → xuất hiện 3-4 windows liên tiếp
- Background species → present suốt file
- Rain noise → giảm confidence đều cả file

SSM/LSTM capture được pattern này, clip-level model thì không.
```

### 8.2 LightProtoSSM (Đơn Giản Hóa Để Hiểu)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleBiSSM(nn.Module):
    """
    Bidirectional SSM - phiên bản đơn giản hóa.
    Xử lý sequence 12 windows, capture temporal patterns.
    """
    def __init__(self, d_model: int = 128, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_model, -1))
        )
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Input/output projections
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Conv for local context
        self.conv = nn.Conv1d(d_model, d_model, 4, padding=3, groups=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model) → (B, T, d_model)"""
        B, T, D = x.shape
        
        # Gated projection
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        
        # Local conv
        x_conv = F.silu(self.conv(x_ssm.transpose(1, 2))[:, :, :T].transpose(1, 2))
        
        # SSM
        dt = F.softplus(self.dt_proj(x_conv))  # (B, T, D)
        A = -torch.exp(self.A_log)              # (D, d_state)
        B_mat = self.B_proj(x_conv)             # (B, T, d_state)
        C_mat = self.C_proj(x_conv)             # (B, T, d_state)
        
        # Sequential state update
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(T):
            dA = torch.exp(A[None] * dt[:, t, :, None])      # (B, D, d_state)
            dB = dt[:, t, :, None] * B_mat[:, t, None, :]     # (B, D, d_state)
            h = h * dA + x[:, t, :, None] * dB
            y = (h * C_mat[:, t, None, :]).sum(-1)            # (B, D)
            ys.append(y)
        
        y_seq = torch.stack(ys, dim=1)  # (B, T, D)
        return y_seq + x * self.D[None, None, :]


class TemporalBirdClassifier(nn.Module):
    """
    Full temporal model: Perch embeddings → species probabilities
    với temporal context across 12 windows.
    """
    def __init__(
        self,
        d_input: int = 1536,    # Perch embedding dim
        d_model: int = 128,
        n_classes: int = 234,
        n_windows: int = 12,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.n_windows = n_windows
        
        # Project embeddings
        self.input_proj = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, n_windows, d_model) * 0.02)
        
        # Bidirectional SSM
        self.ssm_fwd = SimpleBiSSM(d_model)
        self.ssm_bwd = SimpleBiSSM(d_model)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
        # Cross-attention (captures species co-occurrence)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=2, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(d_model)
        
        # Prototype-based classification
        self.prototypes = nn.Parameter(torch.randn(n_classes, d_model) * 0.02)
        self.temp = nn.Parameter(torch.tensor(5.0))
        self.class_bias = nn.Parameter(torch.zeros(n_classes))
        
        # Blend với Perch logits
        self.fusion_alpha = nn.Parameter(torch.zeros(n_classes))
    
    def forward(
        self,
        emb: torch.Tensor,           # (B, T, 1536) Perch embeddings
        perch_logits: torch.Tensor = None  # (B, T, 234) optional
    ) -> torch.Tensor:
        B, T, _ = emb.shape
        
        # Project + positional encoding
        h = self.input_proj(emb) + self.pos_enc[:, :T, :]
        
        # Bidirectional SSM
        res = h
        hf = self.ssm_fwd(h)
        hb = self.ssm_bwd(h.flip(1)).flip(1)
        h = self.merge(torch.cat([hf, hb], dim=-1))
        h = self.norm(h + res)
        
        # Cross-attention
        h_attn, _ = self.cross_attn(h, h, h)
        h = self.cross_norm(h + h_attn)
        
        # Prototype similarity
        h_norm = F.normalize(h, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(h_norm, p_norm.T) * F.softplus(self.temp) + self.class_bias
        
        # Blend với Perch
        if perch_logits is not None:
            alpha = torch.sigmoid(self.fusion_alpha)[None, None, :]
            sim = alpha * sim + (1 - alpha) * perch_logits
        
        return sim  # (B, T, n_classes)
```

---

## 9. PHASE 5 — PSEUDO-LABELING & SEMI-SUPERVISED

### 9.1 PowerTransform Pseudo-Labeling (2025 1st Place Core Innovation)

```
Vấn đề với naive pseudo-labeling:
- Probability collapse: sau mỗi round, probs tập trung về 0.5
- Soft label "mờ" → model học ít discriminative features

Giải pháp: PowerTransform
- p_new = p^γ với γ > 1 (sharpening)
- Giữ discriminative signal qua nhiều rounds
- 4 iterations → +3.2% LB trong 2025!
```

```python
def power_transform_pseudo_labels(
    probs: np.ndarray,    # (N, n_classes) ∈ [0, 1]
    gamma: float = 2.0,   # > 1: sharpen, < 1: soften
    threshold: float = 0.5,
    normalize: bool = False
) -> np.ndarray:
    """
    Apply power transform để sharpen pseudo-labels.
    
    Args:
        probs: Raw prediction probabilities
        gamma: Power factor. 2.0 = sharpen aggressively
        threshold: Only keep predictions above this
        normalize: Normalize sum to 1 per sample
    """
    # Apply power transform
    pseudo = np.power(np.clip(probs, 1e-7, 1.0), gamma)
    
    # Threshold: suppress uncertain predictions
    pseudo[probs < threshold] = 0.0
    
    if normalize:
        row_sums = pseudo.sum(axis=1, keepdims=True) + 1e-9
        pseudo = pseudo / row_sums
    
    return pseudo.astype(np.float32)


class PseudoLabelingPipeline:
    """
    OOF Pseudo-Labeling với 4 rounds.
    Theo approach của 2025 1st place winner.
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        n_rounds: int = 4,
        gamma_schedule: list = [2.0, 1.8, 1.6, 1.4],
        threshold: float = 0.5,
    ):
        self.n_folds = n_folds
        self.n_rounds = n_rounds
        self.gamma_schedule = gamma_schedule
        self.threshold = threshold
    
    def round_1_oof(self, train_data, model_class, train_fn):
        """Round 1: Train trên original data, generate OOF pseudo-labels."""
        from sklearn.model_selection import StratifiedKFold
        
        pseudo_labels = np.zeros((len(train_data), 234))
        
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_data, ...)):
            print(f"Fold {fold+1}/{self.n_folds}")
            
            # Train on fold
            model = model_class()
            model = train_fn(model, train_data[train_idx])
            
            # Generate pseudo-labels for val fold
            val_preds = predict(model, train_data[val_idx])
            pseudo_labels[val_idx] = val_preds
        
        return pseudo_labels
    
    def run(self, train_data, unlabeled_data, model_class, train_fn):
        """Run full pseudo-labeling pipeline."""
        all_pseudo = []
        
        # Round 1: OOF on original data
        pseudo = self.round_1_oof(train_data, model_class, train_fn)
        
        for round_idx in range(1, self.n_rounds):
            gamma = self.gamma_schedule[round_idx]
            print(f"\n🔄 Pseudo-labeling round {round_idx+1}, gamma={gamma}")
            
            # PowerTransform
            sharp_pseudo = power_transform_pseudo_labels(pseudo, gamma=gamma)
            
            # Select confident predictions
            confident_mask = sharp_pseudo.max(axis=1) > self.threshold
            confident_data = unlabeled_data[confident_mask]
            confident_labels = sharp_pseudo[confident_mask]
            
            print(f"   Confident samples: {confident_mask.sum()}/{len(unlabeled_data)}")
            
            # Combine với original training data
            combined_data = np.concatenate([train_data, confident_data])
            combined_labels = np.concatenate([train_labels, confident_labels])
            
            # Retrain
            model = model_class()
            model = train_fn(model, combined_data, combined_labels)
            
            # Update pseudo-labels cho next round
            pseudo = predict(model, unlabeled_data)
            all_pseudo.append(pseudo)
        
        return model, all_pseudo
```

### 9.2 Non-Bird Pipeline Riêng (2025 1st Place)

```python
# Amphibia + Insecta cần pipeline riêng!
# Vì feature distribution khác hoàn toàn với birds

NON_BIRD_SPECIES = {
    "Amphibia": [...],   # 35 species
    "Insecta": [...],    # 28 sonotypes  
    "Mammalia": [...],   # 8 species
    "Reptilia": [...],   # 1 species
}

def build_non_bird_dataset():
    """
    Thu thập thêm data cho non-bird taxa từ XC/iNat.
    2025 1st place dùng 17,197 extra non-bird recordings!
    """
    # Download từ Xeno-canto API (nếu được phép)
    # Filter theo taxonomy class
    # Balance với bird data
    pass


class NonBirdSEDModel(nn.Module):
    """
    Dedicated model cho Amphibia + Insecta.
    Khác bird model:
    - Lower frequency range (frogs: 100-8000Hz)
    - Different temporal patterns (continuous calls)
    - Cần fine-tune riêng
    """
    def __init__(self, n_non_bird_classes: int, backbone: str = "tf_efficientnet_b0"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=1)
        # ... non-bird specific head
```

---

## 10. PHASE 6 — POST-PROCESSING STACK

### 10.1 Bayesian Prior (Site × Hour)

```python
def build_prior_tables(meta_df: pd.DataFrame, Y_labels: np.ndarray) -> dict:
    """
    Build 3-tier Bayesian prior: global → site+hour → site×hour.
    
    Ý tưởng: Nếu biết location + time, ta có prior về
    loài nào có khả năng xuất hiện cao.
    Pantanal có temporal patterns rõ (dawn chorus, nocturnal species...).
    """
    eps = 1e-4
    
    # Global prior
    global_p = Y_labels.mean(axis=0).astype(np.float32)
    global_p = np.clip(global_p, eps, 1 - eps)
    
    # Site prior (với Bayesian shrinkage)
    site_keys = sorted(meta_df["site"].dropna().unique())
    site_p = {}
    for site in site_keys:
        mask = meta_df["site"].astype(str) == str(site)
        n = mask.sum()
        w = n / (n + 8.0)  # Shrinkage: ít data → lean toward global
        site_p[site] = w * Y_labels[mask].mean(0) + (1 - w) * global_p
    
    # Hour prior
    hour_p = {}
    for h in range(24):
        mask = meta_df["hour_utc"].astype(int) == h
        n = mask.sum()
        if n > 0:
            w = n / (n + 8.0)
            hour_p[h] = w * Y_labels[mask].mean(0) + (1 - w) * global_p
    
    # Joint site × hour prior (tighter shrinkage = 4)
    sh_p = {}
    for site in site_keys:
        for h in range(24):
            mask = (meta_df["site"].astype(str) == str(site)) & \
                   (meta_df["hour_utc"].astype(int) == h)
            n = mask.sum()
            if n > 0:
                w = n / (n + 4.0)  # Tighter shrinkage
                hour_prior = hour_p.get(h, global_p)
                sh_p[(site, h)] = w * Y_labels[mask].mean(0) + (1 - w) * hour_prior
    
    return {
        "global": global_p,
        "site": site_p,
        "hour": hour_p,
        "site_hour": sh_p,
    }


def apply_prior_to_logits(
    logits: np.ndarray,     # (N, 234) raw logits
    sites: np.ndarray,      # (N,) site IDs
    hours: np.ndarray,      # (N,) UTC hours
    prior_tables: dict,
    lambda_prior: float = 0.4
) -> np.ndarray:
    """
    Inject prior knowledge vào logits.
    
    Formula: logit_adj = logit + λ * log(prior / (1 - prior))
    """
    N = len(logits)
    adjusted = logits.copy()
    
    global_p = prior_tables["global"]
    global_logit = np.log(global_p / (1 - global_p + 1e-8))
    
    for i in range(N):
        site = str(sites[i])
        hour = int(hours[i]) % 24
        
        # Get best prior (site×hour > site > hour > global)
        if (site, hour) in prior_tables["site_hour"]:
            p = prior_tables["site_hour"][(site, hour)]
        elif site in prior_tables["site"]:
            p = prior_tables["site"][site]
        elif hour in prior_tables["hour"]:
            p = prior_tables["hour"][hour]
        else:
            p = global_p
        
        p = np.clip(p, 1e-4, 1 - 1e-4)
        prior_logit = np.log(p / (1 - p))
        adjusted[i] += lambda_prior * (prior_logit - global_logit)
    
    return adjusted.astype(np.float32)
```

### 10.2 Temporal Smoothing

```python
def temporal_continuity_smooth(
    probs: np.ndarray,     # (N, 234) trong đó N = n_files × 12
    n_windows: int = 12,
    alpha: float = 0.20,
    kernel: str = "t_dist"  # "gaussian" or "t_dist"
) -> np.ndarray:
    """
    Smooth predictions across temporal windows.
    Lý do: species xuất hiện liên tục, không ngắt quãng đột ngột.
    """
    N, C = probs.shape
    assert N % n_windows == 0
    
    view = probs.reshape(-1, n_windows, C)  # (n_files, 12, 234)
    result = view.copy()
    
    if kernel == "t_dist":
        # Fat-tailed t-distribution kernel (35-second context)
        # Robust hơn Gaussian với outliers
        offsets = np.arange(-3, 4, dtype=np.float32)
        weights = (1.0 + (offsets / 1.20) ** 2 / 2.0) ** (-1.5)
        weights /= weights.sum()
    else:
        # Simple Gaussian
        weights = np.exp(-0.5 * np.arange(-3, 4) ** 2)
        weights /= weights.sum()
    
    for t in range(n_windows):
        neighbors = []
        for offset, w in zip(range(-3, 4), weights):
            t_idx = min(max(t + offset, 0), n_windows - 1)
            neighbors.append(w * view[:, t_idx, :])
        
        smoothed = sum(neighbors)
        conf = view[:, t, :].max(axis=-1, keepdims=True)
        alpha_adaptive = alpha * (1.0 - conf)
        result[:, t, :] = (1 - alpha_adaptive) * view[:, t, :] + alpha_adaptive * smoothed
    
    return result.reshape(N, C)


def file_confidence_scaling(
    probs: np.ndarray,
    n_windows: int = 12,
    top_k: int = 2,
    power: float = 0.4
) -> np.ndarray:
    """
    Scale probabilities based on file-level confidence.
    High confidence file → boost all predictions.
    Low confidence file → suppress.
    """
    N, C = probs.shape
    view = probs.reshape(-1, n_windows, C)
    
    # Top-2 mean across windows as confidence estimate
    sorted_v = np.sort(view, axis=1)
    top_k_mean = sorted_v[:, -top_k:, :].mean(axis=1, keepdims=True)
    
    return (view * np.power(top_k_mean, power)).reshape(N, C)


def apply_topn_postprocessing(
    probs: np.ndarray,
    n_windows: int = 12,
    n: int = 1
) -> np.ndarray:
    """
    TopN postprocessing (2025 2nd place: +1.1% LB).
    Multiply each window's prediction by the top-N max across all windows.
    Boosts consistent species, suppresses spurious detections.
    """
    N, C = probs.shape
    view = probs.reshape(-1, n_windows, C)
    
    # Top-N max per class across all windows in file
    sorted_by_window = np.sort(view, axis=1)  # (n_files, 12, 234)
    topn_max = sorted_by_window[:, -n:, :].mean(axis=1, keepdims=True)  # (n_files, 1, 234)
    
    result = view * topn_max
    return result.reshape(N, C)
```

### 10.3 Sonotype Mirroring

```python
SONOTYPE_GROUPS = [
    ("47158son15", "47158son16"),
    ("47158son09", "47158son12"),
    ("47158son02", "47158son14"),
    ("47158son13", "47158son21", "47158son22", "47158son23"),
]

def apply_sonotype_mirroring(
    probs: np.ndarray,
    primary_labels: list,
    sonotype_groups: list = SONOTYPE_GROUPS
) -> np.ndarray:
    """
    Sonotypes trong cùng nhóm có spectrogram tương tự.
    → Max-pool predictions để đảm bảo nhất quán.
    """
    col_to_idx = {l: i for i, l in enumerate(primary_labels)}
    result = probs.copy()
    
    for group in sonotype_groups:
        valid_idx = [col_to_idx[s] for s in group if s in col_to_idx]
        if len(valid_idx) < 2:
            continue
        
        group_max = probs[:, valid_idx].max(axis=1, keepdims=True)
        result[:, valid_idx] = group_max
    
    return result
```

---

## 11. PHASE 7 — ENSEMBLE STRATEGY

### 11.1 Rank-Percentile Blending

```python
import pandas as pd
import numpy as np

def rank_percentile_blend(
    probs_list: list,      # List of (N, 234) probability arrays
    weights: list,         # Corresponding weights
    eps: float = 1e-5
) -> np.ndarray:
    """
    Rank percentile ensemble thay vì simple average.
    Tốt hơn khi các model có calibration khác nhau.
    
    Ví dụ: 60% ProtoSSM + 40% SED
    """
    assert len(probs_list) == len(weights)
    assert abs(sum(weights) - 1.0) < 1e-6
    
    rank_arrays = []
    for probs in probs_list:
        clipped = np.clip(probs, eps, 1 - eps)
        ranked = pd.DataFrame(clipped).rank(axis=0, pct=True).to_numpy(np.float32)
        rank_arrays.append(ranked)
    
    # Weighted sum of ranks
    blended = sum(w * r for w, r in zip(weights, rank_arrays))
    return blended.astype(np.float32)


def optuna_blend_weights(
    val_probs_list: list,   # OOF predictions
    val_labels: np.ndarray,
    n_trials: int = 200
) -> list:
    """
    Dùng Optuna để tìm optimal ensemble weights.
    Như VSydorskyy (2025 2nd place) với 15 models.
    """
    import optuna
    from sklearn.metrics import roc_auc_score
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        # Sample weights (simplex)
        n = len(val_probs_list)
        raw_weights = [trial.suggest_float(f"w{i}", 0.0, 1.0) for i in range(n)]
        weights = np.array(raw_weights) / sum(raw_weights)
        
        blended = rank_percentile_blend(val_probs_list, weights.tolist())
        
        keep = val_labels.sum(axis=0) > 0
        auc = roc_auc_score(val_labels[:, keep], blended[:, keep], average="macro")
        return auc
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_weights = [study.best_params[f"w{i}"] for i in range(len(val_probs_list))]
    best_weights = np.array(best_weights) / sum(best_weights)
    
    print(f"Best weights: {best_weights}")
    print(f"Best AUC: {study.best_value:.4f}")
    
    return best_weights.tolist()
```

### 11.2 Model Soup (Free Performance Boost)

```python
def model_soup(checkpoint_paths: list, model_class, weight: str = "uniform") -> dict:
    """
    Average weights của nhiều checkpoints.
    2025 1st và 3rd place đều dùng kỹ thuật này.
    """
    import torch
    
    checkpoints = [torch.load(p, map_location="cpu") for p in checkpoint_paths]
    
    soup_state = {}
    for key in checkpoints[0].keys():
        if weight == "uniform":
            soup_state[key] = torch.stack([ckpt[key].float() for ckpt in checkpoints]).mean(0)
        elif isinstance(weight, list):
            w_sum = sum(weight)
            soup_state[key] = sum(
                w / w_sum * ckpt[key].float() 
                for w, ckpt in zip(weight, checkpoints)
            )
    
    model = model_class()
    model.load_state_dict(soup_state)
    return model
```

---

## 12. CHIẾN LƯỢC CPU INFERENCE TỐI ƯU

### 12.1 Timeline Budget (90 phút)

```
~600 test files × 12 windows = 7,200 windows total

Budget allocation:
├── ONNX Perch inference:        ~5-8 min  (batch=16, parallel IO)
├── LightProtoSSM training:      ~1-2 min  (40 epochs trên 59 files)
├── MLP Probes training:         ~1 min    (58 active classes)
├── SSM/ProtoSSM inference:      ~1 min    
├── SED 5-fold ONNX inference:   ~5-8 min
├── Post-processing:             ~1 min
└── Ensemble + save:             ~1 min
                                 ─────────
Total:                           ~15-22 min  ✅ Well within 90 min

ONNX là key: TF SavedModel ~150x slower hơn ONNX trên CPU!
```

### 12.2 Batch Processing

```python
import concurrent.futures

def run_perch_batched(
    paths: list,
    session,
    input_name: str,
    out_map: dict,
    batch_size: int = 16,
    n_io_workers: int = 4,
):
    """
    Tối ưu Perch inference với:
    1. Batched inference (16 files cùng lúc)
    2. Parallel IO (4 threads đọc audio trước)
    3. Prefetching (đọc batch tiếp khi đang infer)
    """
    paths = [Path(p) for p in paths]
    n_rows = len(paths) * N_WINDOWS
    
    results = {
        "row_ids": np.empty(n_rows, dtype=object),
        "sites": np.empty(n_rows, dtype=object),
        "hours": np.zeros(n_rows, dtype=np.int16),
        "scores": np.zeros((n_rows, N_CLASSES), dtype=np.float32),
        "embs": np.zeros((n_rows, 1536), dtype=np.float32),
    }
    
    wr = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_io_workers) as io_pool:
        # Prefetch first batch
        next_batch = paths[:batch_size]
        future_audio = [io_pool.submit(read_60s, p) for p in next_batch]
        
        for start in tqdm(range(0, len(paths), batch_size), desc="Perch"):
            batch_paths = next_batch
            batch_n = len(batch_paths)
            batch_audio = [f.result() for f in future_audio]
            
            # Prefetch next batch WHILE processing current
            next_start = start + batch_size
            if next_start < len(paths):
                next_batch = paths[next_start:next_start + batch_size]
                future_audio = [io_pool.submit(read_60s, p) for p in next_batch]
            
            # Prepare batch
            x = np.stack([
                audio.reshape(N_WINDOWS, WINDOW_SAMPLES) 
                for audio in batch_audio
            ]).reshape(-1, WINDOW_SAMPLES)  # (B*12, 160000)
            
            # ONNX inference
            outs = session.run(None, {input_name: x})
            logits = outs[out_map["label"]].astype(np.float32)   # (B*12, 14795)
            embs = outs[out_map["embedding"]].astype(np.float32) # (B*12, 1536)
            
            # Store results
            br = wr
            for bi, (path, audio) in enumerate(zip(batch_paths, batch_audio)):
                meta = parse_fname(path.name)
                stem = path.stem
                results["row_ids"][wr:wr+N_WINDOWS] = [f"{stem}_{t}" for t in range(5, 65, 5)]
                results["sites"][wr:wr+N_WINDOWS] = meta["site"]
                results["hours"][wr:wr+N_WINDOWS] = meta["hour_utc"]
                wr += N_WINDOWS
            
            results["scores"][br:wr, MAPPED_POS] = logits[:, MAPPED_BC_IDX]
            results["embs"][br:wr] = embs
    
    return results
```

### 12.3 ONNX Optimization Tips

```python
# Tips để tối ưu ONNX inference trên Kaggle CPU

# 1. Số threads
so = ort.SessionOptions()
so.intra_op_num_threads = 4   # Kaggle có 4 cores
so.inter_op_num_threads = 1
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 2. Batch size optimal
# Quá nhỏ: overhead overhead nhiều
# Quá lớn: memory pressure
OPTIMAL_BATCH = 16  # files (= 16 × 12 = 192 windows)

# 3. Data type
# Luôn dùng float32, không float64
x = x.astype(np.float32)

# 4. Memory pre-allocation
# Pre-allocate output arrays thay vì concatenate
scores_all = np.zeros((n_total_windows, 234), dtype=np.float32)
# Fill in-place:
scores_all[start:end] = batch_scores
```

---

## 13. LỘ TRÌNH THỰC CHIẾN TUẦN-THEO-TUẦN

### Giai Đoạn 1: Foundation (Tuần 1-2)

```
📅 Tuần 1:
□ Fork notebook public (LB 0.946) và chạy để hiểu
□ EDA: phân tích taxonomy, soundscape labels, data quality
□ Setup local dev environment
□ Đọc: BirdCLEF 2025 top solutions (các link trong document)
□ Chạy thành công notebook public → base score

📅 Tuần 2:
□ Implement SED training script (EfficientNet-B2)
□ Implement augmentation: MixUp audio, SpecAugment
□ Train 5-fold SED trên train_audio + soundscapes
□ Target: val AUC > 0.85 trên labeled soundscapes
□ Export sang ONNX, verify speed
```

### Giai Đoạn 2: Core Improvements (Tuần 3-4)

```
📅 Tuần 3:
□ Pseudo-labeling round 1: dùng SED model predict train_soundscapes
□ PowerTransform với γ=2.0 → sharpen predictions
□ Retrain với combined data
□ Implement SoftAUCLoss
□ Target: val AUC > 0.90

📅 Tuần 4:
□ Pseudo-labeling round 2-3 (OOF approach)
□ Backbone thử nghiệm: B4, EfficientNetV2-S
□ Non-bird pipeline riêng (Amphibia + Insecta)
□ Cải thiện ProtoSSM: thêm site/hour embeddings
□ Target: LB > 0.92
```

### Giai Đoạn 3: Advanced Techniques (Tuần 5-6)

```
📅 Tuần 5:
□ Multi-seed ensemble (3+ seeds)
□ Model Soup (last 3 checkpoints)
□ Optuna cho ensemble weights
□ Background noise augmentation (ESC-50)
□ Target: LB > 0.94

📅 Tuần 6:
□ Full pseudo-labeling với 4 rounds
□ Tinh chỉnh post-processing stack
□ 5-gate ensemble blend
□ OOF calibration + per-class thresholds
□ Target: LB > 0.95
```

### Giai Đoạn 4: Polish & Submit (Tuần 7-8)

```
📅 Tuần 7:
□ CPU runtime optimization (< 60 min để có margin)
□ ONNX export tất cả models
□ Verify submission format
□ Ablation studies: measure contribution của từng component
□ Target: LB > 0.955

📅 Tuần 8 (Final):
□ Final ensemble với best submissions
□ Diversity-based model selection
□ Submit 2+ diverse submissions
□ Backup submission plan (simpler but robust)
□ Working note draft (optional $2,500 prize)
```

---

## 14. CHECKLIST TRƯỚC KHI SUBMIT

```
PRE-SUBMISSION CHECKLIST:

Audio Processing:
□ SR = 32,000 Hz (match test data)
□ Pad audio < 60s, truncate > 60s
□ Windows: 12 × 5s = 60s
□ row_id format: {filename_without_ext}_{end_second}
  Ví dụ: BC2026_Test_0001_S05_20250227_010002_20 (0:15-0:20)

Model & Inference:
□ Tất cả models đã export sang ONNX
□ Verify ONNX output matches PyTorch output (diff < 1e-4)
□ Test runtime trên Kaggle kernel (target: < 60 min)
□ No internet access trong submission: mọi model pre-attached
□ File named exactly: submission.csv

Submission Format:
□ Columns: row_id + 234 species columns (đúng thứ tự từ sample_submission.csv)
□ Probabilities ∈ [0, 1]
□ Số rows = n_test_files × 12
□ Kiểm tra: không có NaN, không có Inf

Code Quality:
□ Không import packages bị cấm
□ Tất cả paths hardcoded hoặc tìm dynamically
□ Error handling cho missing files
□ Memory management: del + gc.collect() sau heavy operations

Final Checks:
□ Chạy notebook từ đầu tới cuối, kernel restart sạch
□ Verify output trên dry-run (dùng train_soundscapes)
□ Compare với sample_submission schema
□ Submit 1 ngày trước deadline (dự phòng)
```

---

## 15. CODE TEMPLATES & SNIPPETS

### 15.1 Complete Training Dataset Class

```python
import torch
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import numpy as np
import pandas as pd

class BirdCLEFDataset(Dataset):
    """
    Dataset cho BirdCLEF training.
    Hỗ trợ cả XC/iNat recordings và labeled soundscapes.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        primary_labels: list,
        n_mels: int = 128,
        duration: float = 5.0,
        sr: int = 32000,
        augment: bool = True,
        augmentor: AudioAugmentation = None,
        noise_paths: list = None,
    ):
        self.df = df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.label_to_idx = {l: i for i, l in enumerate(primary_labels)}
        self.n_classes = len(primary_labels)
        self.n_mels = n_mels
        self.duration = duration
        self.sr = sr
        self.n_samples = int(duration * sr)
        self.augment = augment
        self.augmentor = augmentor or AudioAugmentation(sr)
        self.noise_paths = noise_paths or []
    
    def __len__(self):
        return len(self.df)
    
    def _load_audio(self, path: Path) -> np.ndarray:
        """Load và preprocess audio."""
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
        
        # Random crop 5 seconds
        if len(y) > self.n_samples:
            start = np.random.randint(0, len(y) - self.n_samples)
            y = y[start:start + self.n_samples]
        else:
            y = np.pad(y, (0, self.n_samples - len(y)))
        
        return y
    
    def _build_label(self, row: pd.Series) -> np.ndarray:
        """Build multi-hot label vector."""
        label = np.zeros(self.n_classes, dtype=np.float32)
        
        # Primary label
        if row.get("primary_label") in self.label_to_idx:
            label[self.label_to_idx[row["primary_label"]]] = 1.0
        
        # Secondary labels
        if pd.notna(row.get("secondary_labels", "")):
            for sec in str(row.get("secondary_labels", "")).split(" "):
                sec = sec.strip()
                if sec in self.label_to_idx:
                    label[self.label_to_idx[sec]] = 1.0
        
        return label
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Load audio
        audio_path = self.audio_dir / row["filename"]
        y = self._load_audio(audio_path)
        
        # Build label
        label = self._build_label(row)
        
        # Augmentation
        if self.augment:
            # Random time shift
            if np.random.random() < 0.3:
                y = self.augmentor.time_shift(y)
            
            # Random filtering
            if np.random.random() < 0.3:
                y = self.augmentor.random_filtering(y)
            
            # Background noise
            if self.noise_paths and np.random.random() < 0.5:
                noise_path = np.random.choice(self.noise_paths)
                noise, _ = sf.read(str(noise_path), dtype="float32")
                if noise.ndim == 2:
                    noise = noise.mean(axis=1)
                y = self.augmentor.add_background_noise(y, noise)
        
        # Mel spectrogram
        mel = compute_mel_spectrogram(y, n_mels=self.n_mels)
        
        # SpecAugment (on spectrogram)
        if self.augment:
            mel = self.augmentor.spec_augment(mel)
        
        return {
            "audio": torch.tensor(mel, dtype=torch.float32).unsqueeze(0),  # (1, n_mels, T)
            "labels": torch.tensor(label, dtype=torch.float32),
            "filename": str(row["filename"]),
        }
```

### 15.2 Evaluation Helper

```python
from sklearn.metrics import roc_auc_score

def macro_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Competition metric: macro ROC-AUC, skip classes với 0 true positives.
    """
    keep = y_true.sum(axis=0) > 0
    if keep.sum() == 0:
        return 0.0
    return roc_auc_score(y_true[:, keep], y_score[:, keep], average="macro")


def per_class_auc(y_true: np.ndarray, y_score: np.ndarray, class_names: list) -> pd.DataFrame:
    """Per-class AUC để debug."""
    results = []
    for i, name in enumerate(class_names):
        if y_true[:, i].sum() > 0:
            auc = roc_auc_score(y_true[:, i], y_score[:, i])
            results.append({"class": name, "auc": auc, "n_pos": int(y_true[:, i].sum())})
    
    df = pd.DataFrame(results).sort_values("auc")
    print(f"Worst 10 classes:\n{df.head(10)}")
    print(f"\nBest 10 classes:\n{df.tail(10)}")
    print(f"\nOverall macro AUC: {df['auc'].mean():.4f}")
    return df
```

### 15.3 Submission Generation

```python
def generate_submission(
    test_paths: list,
    perch_session,
    input_name: str,
    out_map: dict,
    proto_model,
    sed_sessions: list,
    prior_tables: dict,
    primary_labels: list,
    output_path: str = "submission.csv",
) -> pd.DataFrame:
    """Full submission generation pipeline."""
    
    print("🦜 Generating submission...")
    
    # Step 1: Perch inference
    print("Step 1/5: Perch inference...")
    meta_te, sc_te, emb_te = run_perch_batched(
        test_paths, perch_session, input_name, out_map
    )
    
    # Step 2: Apply prior
    print("Step 2/5: Applying priors...")
    sc_te_prior = apply_prior_to_logits(
        sc_te, meta_te["site"].values, meta_te["hour_utc"].values,
        prior_tables
    )
    
    # Step 3: ProtoSSM inference
    print("Step 3/5: ProtoSSM inference...")
    proto_scores = run_proto_inference(proto_model, emb_te, sc_te_prior, meta_te)
    
    # Step 4: SED inference
    print("Step 4/5: SED inference...")
    sed_scores = run_sed_inference(sed_sessions, test_paths)
    
    # Step 5: Ensemble + post-processing
    print("Step 5/5: Ensemble & post-processing...")
    
    # Rank percentile blend
    final = rank_percentile_blend([proto_scores, sed_scores], weights=[0.60, 0.40])
    
    # Post-processing
    final = apply_per_class_thresholds(final, PER_CLASS_THRESHOLDS)
    final = temporal_continuity_smooth(final)
    final = file_confidence_scaling(final)
    final = apply_topn_postprocessing(final)
    final = apply_sonotype_mirroring(final, primary_labels)
    
    # Rare-class suppression
    # ...
    
    # Build submission
    sub = pd.DataFrame(final, columns=primary_labels)
    sub.insert(0, "row_id", meta_te["row_id"].values)
    sub.to_csv(output_path, index=False)
    
    print(f"✅ Submission saved: {output_path}, shape={sub.shape}")
    return sub
```

---

## 16. TÀI LIỆU THAM KHẢO

### Papers & Resources

```
BirdCLEF:
1. BirdCLEF 2025 1st Place Write-up (Nikita Babych)
   - PowerTransform Pseudo-labeling
   - SoftAUCLoss
   - Multi-iterative Noisy Student

2. BirdCLEF 2025 2nd Place Write-up (VSydorskyy & Gonçalves)
   - OOF Pseudo-labeling
   - Audio MixUp
   - TopN postprocessing

Models:
3. Google Perch v2: "Self-Supervised Learning for Bird Sound Classification"
4. BirdNET: "A deep learning solution for avian diversity monitoring"
5. EfficientNet: Tan & Le, 2019
6. Mamba/SSM: Gu & Dao, 2023 "Mamba: Linear-Time Sequence Modeling"

Techniques:
7. SpecAugment: Park et al., 2019
8. MixUp: Zhang et al., 2017
9. StochasticDepth: Huang et al., 2016
10. SWA: Izmailov et al., 2018
11. Model Soup: Wortsman et al., 2022
```

### Kaggle Resources

```
Public Notebooks:
- LB 0.946: "ONNX Perch + Sequence Modeling & SED Blend" (Imaad Mahmood)
- Tucker Arrants SED: bc2026-distilled-sed-public
- Perch ONNX: rishikeshjani/perch-onnx-for-birdclef-2026

Discussions:
- Competition hosts on test data statistics
- Export models for Raven Intelligence (ONNX/TorchScript)
- Domain gap analysis reports

Reports (competition forum):
- BirdCLEF2026_Analysis_Report.html (EDA)
- sed_improvement_plan.html (SED roadmap)
- discovery_report_20260319.html
- postproc_report.html
- clap_domain_bridge_design_20260330.html
```

### Score Progression Target

```
Baseline (public notebook fork):       0.946
+ Better SED training (B2/B4 + SWA):  +0.008  → 0.954
+ PowerTransform pseudo-labeling:      +0.010  → 0.964
+ Non-bird dedicated pipeline:         +0.005  → 0.969
+ Extended pretrain (819K BirdCLEF):   +0.008  → 0.977
+ Multi-round pseudo (4 rounds):       +0.005  → 0.982
+ OOF calibration + Optuna blend:      +0.003  → 0.985

Top 1-3 estimate: ~0.97-0.98
```

---

## 📊 TỔNG KẾT — CÁC ĐÒN QUYẾT ĐỊNH

```
Rank 1 Contribution → Implement Priority:

Priority 1 (Highest Impact):
  ✅ ONNX Perch v2 (already in public notebook)
  🎯 SoftAUCLoss thay BCE → +2-3% LB
  🎯 Audio MixUp → +3-4% Private LB (2025 2nd place confirmed!)
  🎯 PowerTransform pseudo-labels × 4 rounds → +3% LB

Priority 2 (Medium Impact):
  🎯 EfficientNet-B4/V2-S backbone thay B0 → +1-2%
  🎯 Non-bird dedicated pipeline → +0.5-1%
  🎯 Model Soup (last 3 checkpoints) → free +0.3-0.5%
  🎯 TopN postprocessing → confirmed +1.1% LB
  🎯 Background noise augmentation → +0.5-1%

Priority 3 (Refinement):
  🎯 Optuna ensemble weights (15+ models)
  🎯 Per-class threshold tuning
  🎯 StochasticDepth regularization
  🎯 OOF calibration + isotonic regression

The Golden Rule:
  "Pseudo-labeling với PowerTransform là sự khác biệt 
   giữa top 10% và top 1%"
```

---

*Document được tổng hợp từ: Competition overview, EDA reports, BirdCLEF 2025 top solutions, public LB 0.946 notebook analysis. Cập nhật: Tháng 5/2026.*

*🦜 Good luck! Pantanal's biodiversity is counting on you.*
