<div align="center">

# SeetaPsych Hertz

<img src="website/public/media/tinyhr-logo.png" width="460" alt="SeetaPsych Hertz logo">

### Recover the pulse from subtle skin-color changes

TinyHR estimates heart rate without contact by recovering a pulse waveform from subtle
facial skin-color variations associated with pulsatile blood-volume changes in RGB video.

[English](README.md) | [简体中文](README_CN.md)

[Introduction](#introduction) · [Installation](#installation) · [Demo](#demo) · [Datasets](#training-data) · [Benchmark](#model-size-and-latency) · [Technical Report](website/public/downloads/tinyhr-technical-report.pdf)

[![TinyHR demo showing facial video, predicted pulse waveform, and heart-rate estimates](website/public/media/tinyhr-demo.gif)](website/public/media/demo-full.mp4)

*Watch TinyHR estimate a pulse waveform and heart rate from facial video. Click for the full demo.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-2563D8?logo=python&logoColor=white)](pyproject.toml)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU%20%7C%20GPU-091D31?logo=onnx&logoColor=white)](seetapsych_hertz/modules/tiny-hr.yml)
[![TinyHR](https://img.shields.io/badge/TinyHR-82%2C177_params-FB5F14)](seetapsych_hertz/modules/tiny-hr.yml)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-75E5C9)](LICENSE)

</div>

## Introduction

SeetaPsych Hertz provides heart-rate estimation modules for the
[SeetaPsych](https://github.com/seetapsych/seetapsych-lib) ecosystem. TinyHR is a lightweight
convolutional model for remote photoplethysmography (rPPG). It encodes neighboring-frame
differences and aggregates temporal features at multiple scales to predict one pulse-waveform
sample per input frame. Heart rate is then estimated from the dominant frequency of the
band-pass-filtered waveform using Welch power spectral density (PSD) estimation.

The exported ONNX model contains **82,177 parameter elements** and occupies approximately
**381 KiB**. The training subsets reported across four datasets sum to **1,288 subjects**
and **6,307 videos**. The streaming module uses a 160-frame input window and a default
update interval of 1.0 s once sufficient valid face frames have been collected.

## Major Features

| | Feature | Why it matters |
|---|---|---|
| 🌍 | **Multi-source training data** | Training subsets from four rPPG datasets total 1,288 subjects and 6,307 videos, spanning multiple recording conditions. |
| 🪶 | **Compact convolutional model** | Approximately 82K exported parameters and a 381 KiB ONNX file reduce model-storage requirements for edge deployment. |
| ⚡ | **Low computational latency** | A previous local CPU benchmark recorded a mean of 132 ms; the streaming module has a separate default update interval of 1.0 s. |
| 📹 | **Contactless measurement** | A regular RGB camera provides the facial video input; no wearable sensor is required for the estimation pipeline. |
| 📈 | **Waveform-based estimation** | TinyHR predicts an rPPG waveform, then estimates heart rate by signal processing; the predicted waveform remains available for analysis. |
| 🧩 | **SeetaPsych integration** | Ready-made modules support video files and live video streams through the SeetaPsych pipeline. |

## Key Numbers

| Training datasets | Training subjects (sum) | Training videos | ONNX parameters | Model size | VIPL-HR V1 test MAE |
|---:|---:|---:|---:|---:|---:|
| **4** | **1,288** | **6,307** | **82,177** | **381 KiB** | **3.88 BPM** |

## Demo

The demonstration presents the detected face, predicted rPPG waveform, and heart-rate
estimate together. It illustrates the processing pipeline; accuracy is reported
separately in the evaluation section.

## Training Data

TinyHR was trained on subsets of four rPPG datasets. The counts below describe the
training data used in the technical report, with an added total row; they are not the
full sizes of the original datasets.

| Dataset | Training subjects | Training videos | Usage |
|---|---:|---:|---|
| VIPL-HR V1 | 85 | 1,883 | Training |
| VIPL-HR V2 | 500 | 2,498 | Training |
| V4V | 103 | 726 | Training |
| MCD-rPPG | 600 | 1,200 | Training; front-facing videos only |
| **Total** | **1,288** | **6,307** | **Four-source training corpus** |

The VIPL-HR V1 test split contains **22 subjects and 485 videos** and, according to the
report, was excluded from training. The subject total above is the sum of the four
reported training-subset counts.

| Training configuration | Setting |
|---|---|
| Batch size | 4 |
| Initial learning rate | 0.005 |
| Learning-rate scheduler | OneCycleLR |
| Input clip | 160 RGB face crops, each resized to 128 × 128 pixels |

Source: [TinyHR technical report](website/public/downloads/tinyhr-technical-report.pdf), page 1 (input) and pages 6-7 (training data and configuration).

## Model Size and Latency

| Measurement | Result | Interpretation |
|---|---:|---|
| Exported ONNX parameters | **82,177** | Initializer-element count after export and convolution/normalization fusion; not the pre-export trainable-parameter count |
| ONNX file size | **≈ 381 KiB** | 390,150 bytes; SHA-256 matches the checksum declared in `tiny-hr.yml` |
| Reference processing latency | **132 ms (mean)** | Previous local measurement including input normalization, ONNX inference, waveform normalization, and heart-rate post-processing |
| Reference latency distribution | **125 ms (median); 164 ms (P95)** | 50 timed runs after warm-up with ONNX Runtime CPUExecutionProvider |
| Input observation duration | **≈ 5.3 s at 30 FPS** | Time to collect 160 valid face frames; the first result also depends on update scheduling and computation |
| Rolling update interval | **1.0 s default** | Timestamp-based update requests; approximately 30 frames per interval at 30 FPS |

**Measurement scope.** The previous local benchmark used ONNX Runtime 1.30.0
CPUExecutionProvider and 50 timed runs after warm-up, excluding video acquisition and
face detection. These are implementation measurements, separate from the report's
accuracy evaluation; latency depends on the runtime environment and processing-window length.

The 160-frame input, computation time, and update interval describe different stages.
The streaming implementation accumulates predicted waveform segments over up to 20 s
for heart-rate estimation, so a 1.0 s update interval does not imply a 1.0 s response to
physiological changes. The compact model and rolling estimates support prototyping in
human-computer interaction, affective computing, and contactless monitoring research.

## How TinyHR Works

[![TinyHR architecture and inference flow](website/public/media/tinyhr-flowchart.png)](website/public/downloads/tinyhr-flowchart.pdf)

*Click the diagram to open the architecture PDF. Some diagram labels differ from the report's text: TinyHR uses convolutional MTF processing and non-overlapping spatial patches, as described below.*

| Stage | Module | Function |
|---:|---|---|
| 01 | Frame Difference Fusion Stem | Concatenates four neighboring-frame RGB differences into 12 channels and extracts features through the difference branch; the raw-RGB appearance branch is unused. |
| 02 | Spatial Patch Embedding | A frame-wise 4 × 4 convolution with stride 4 maps 32 × 32 features to a non-overlapping 8 × 8 grid with 32 channels, preserving temporal resolution. |
| 03 | Multi-scale Temporal Feature (MTF) Block | Spatial enhancement and global average pooling feed four parallel temporal-shift branches; pointwise convolutions, a temporal feed-forward network, and residual fusion integrate the features. |
| 04 | Waveform Predictor | Spatial global average pooling and two pointwise convolutions produce one rPPG sample per input frame, retaining the 160-frame temporal resolution. |
| 05 | Signal Processing | The implementation detrends and band-pass-filters the predicted waveform, then estimates heart rate from the dominant Welch PSD peak; streaming uses up to 20 s of accumulated waveform. |

Inference uses a Butterworth band-pass filter with cutoff frequencies of 0.75 and
2.5 Hz (45-150 BPM). Within this configured search band, the frequency of the largest
Welch PSD peak is converted to heart rate:

```text
Heart rate (BPM) = 60 × dominant frequency (Hz)
```

The technical report defines three training terms: negative Pearson correlation
(`L_time = -ρ`) for temporal waveform agreement; cross-entropy (`L_CE`) over 45-149 BPM
frequency bins, supervised by the reference blood volume pulse (BVP) spectrum's dominant
bin; and KL divergence (`L_KL`) between Gaussian distributions centered on the reference
and predicted spectral peaks. Their weights are 0.2, 1.0, and 1.0:

```text
L = 0.2 L_time + L_CE + L_KL
```

## Evaluation

| Dataset | Test split | Protocol | MAE |
|---|---|---|---:|
| VIPL-HR V1 | 22 subjects · 485 videos | Held out from training | **3.88 BPM** |

The technical report gives a heart-rate mean absolute error (MAE) of 3.88 BPM on this
held-out split. The result characterizes this evaluation protocol; it does not establish
cross-dataset generalization or clinical measurement accuracy.

## Installation

This project is included in the default SeetaPsych configuration. Download its modules
with:

```bash
seetapsych-manager download
```

For the complete framework workflow, see
[SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

## Usage

### WebUI

```bash
seetapsych-webui --files seetapsych_hertz/modules/tiny-hr.yml
```

### Programmatic Usage

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_hertz/modules/tiny-hr.yml")

pipeline = Pipeline(factory, ...)
pipeline.add_attributes("face/heart_rate")
```

## Module Zoo

| Module | Description | Input modes |
|---|---|---|
| [AdaChrom](seetapsych_hertz/modules/ada-chrom.yml) | Chrominance-based rPPG on adaptive skin ROIs, without a learned pulse estimator | Video stream · video file |
| [Seeta](seetapsych_hertz/modules.inactived/seeta.yml) | Forehead chrominance and FFT-based heart-rate estimation; configuration currently inactive | Video stream · video file |
| [TinyHR](seetapsych_hertz/modules/tiny-hr.yml) | Convolutional rPPG waveform estimation with Welch PSD-based heart-rate post-processing | Video stream · video file |

### TinyHR Parameters

| Name | Type | Default | Description |
|---|---|---:|---|
| `fps` | number | `30` | Sampling rate used for waveform buffering and spectral analysis; must match the effective input frame rate |
| `interval` | number | `1` | Timestamp-based interval between update requests, in seconds; distinct from observation duration and computation time |

### AdaChrom Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `window_samples` | integer | `300` | Analysis-window length in frames; controls the temporal context and the stability/responsiveness trade-off |
| `roi_regions` | `selection[]` | `["skin_b_adaptive_forehead"]` | Skin regions evaluated before valid estimates are fused |

The listed modules provide the
[`face/heart_rate`](https://github.com/seetapsych/seetapsych-attributes#faceheart_rate)
attribute.

## Resources

- [Full recorded demo](website/public/media/demo-full.mp4)
- [TinyHR technical report](website/public/downloads/tinyhr-technical-report.pdf)
- [Architecture diagram](website/public/downloads/tinyhr-flowchart.pdf)
- [Interactive project page source](website/)
- Hugging Face model distribution and interactive demos are planned.

## License

This project is released under the [BSD 3-Clause License](LICENSE).

## Affiliations

<p align="center">
  <a href="https://scholar.google.com/citations?user=HRBTJYYAAAAJ" title="Southeast University"><img src="website/public/media/affiliations/southeast-university.png" alt="Southeast University" height="104" /></a>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://vipl.ict.ac.cn/en/index.html" title="Institute of Computing Technology, Chinese Academy of Sciences"><img src="website/public/media/affiliations/ict-cas.png" alt="Institute of Computing Technology, Chinese Academy of Sciences" height="72" /></a>
</p>
