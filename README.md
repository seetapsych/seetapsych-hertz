<div align="center">

# SeetaPsych Hertz

<img src="website/public/media/tinyhr-logo.png" width="460" alt="SeetaPsych Hertz logo">

### See the pulse. Through video.

Contactless heart-rate estimation from ordinary facial video, powered by a compact rPPG model.

[English](README.md) | [简体中文](README_CN.md)

[Introduction](#introduction) · [Installation](#installation) · [Demo](#demo) · [Datasets](#training-data) · [Benchmark](#model-size-and-latency) · [Technical Report](website/public/downloads/tinyhr-technical-report.pdf)

[![TinyHR demo showing facial video, predicted pulse waveform, and heart-rate estimates](website/public/media/tinyhr-demo.gif)](website/public/media/demo-full.mp4)

*Watch TinyHR turn facial video into a live pulse waveform. Click for the full demo.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-2563D8?logo=python&logoColor=white)](pyproject.toml)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU%20%7C%20GPU-091D31?logo=onnx&logoColor=white)](seetapsych_hertz/modules/tiny-hr.yml)
[![TinyHR](https://img.shields.io/badge/TinyHR-82%2C177_params-FB5F14)](seetapsych_hertz/modules/tiny-hr.yml)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-75E5C9)](LICENSE)

</div>

## Introduction

SeetaPsych Hertz provides heart-rate estimation modules for the
[SeetaPsych](https://github.com/seetapsych/seetapsych-lib) ecosystem. Its TinyHR model
recovers an rPPG waveform from subtle color changes in facial video, then converts the
waveform into heart rate through filtering and spectral analysis.

TinyHR is designed for practical deployment: the published ONNX model has only
**82,177 parameters** and occupies **381 KiB**, while its four-dataset training corpus
covers **1,288 subjects** and **6,307 videos**. Once the initial video window is ready,
the default streaming pipeline produces a new estimate every second.

## Major Features

| | Feature | Why it matters |
|---|---|---|
| 🌍 | **Diverse multi-source training data** | Four rPPG datasets span 1,288 subjects and 6,307 training videos, increasing diversity across people and recording conditions. |
| 🪶 | **82K-parameter TinyHR model** | The 381 KiB ONNX model is small enough for resource-conscious and edge-oriented deployments. |
| ⚡ | **Fast compute and rolling response** | The measured TinyHR pipeline takes about 132 ms on an Apple M4 CPU and refreshes the estimate every 1 second after warm-up. |
| 📹 | **Contactless measurement** | A regular RGB camera provides the facial video input; no wearable sensor is required for the estimation pipeline. |
| 📈 | **Waveform-first inference** | TinyHR predicts an rPPG waveform before deterministic heart-rate estimation, keeping the physiological signal available for inspection. |
| 🧩 | **SeetaPsych integration** | Ready-made modules support video files and live video streams through the SeetaPsych pipeline. |

## Key Numbers

| Training datasets | Training subjects | Training videos | Parameters | Model size | Test MAE |
|---:|---:|---:|---:|---:|---:|
| **4** | **1,288** | **6,307** | **82,177** | **381 KiB** | **3.88 BPM** |

## Demo

The demonstration presents the detected face, predicted BVP waveform, and heart-rate
estimate together. It illustrates the processing pipeline; accuracy is reported
separately in the evaluation section.

## Training Data

TinyHR was trained with four complementary rPPG datasets. The table below reproduces
the dataset table in the supplied technical report.

| Dataset | Subjects | Videos | Usage |
|---|---:|---:|---|
| VIPL-HR V1 | 85 | 1,883 | Training |
| VIPL-HR V2 | 500 | 2,498 | Training |
| V4V | 103 | 726 | Training |
| MCD-rPPG | 600 | 1,200 | Training; front-facing videos only |
| **Total** | **1,288** | **6,307** | **Four-source training corpus** |

The independent VIPL-HR V1 test split contains **22 subjects and 485 videos**. The
report states that this split was used only for evaluation and was excluded from
training.

| Training configuration | Setting |
|---|---|
| Batch size | 4 |
| Initial learning rate | 0.005 |
| Learning-rate scheduler | OneCycleLR |
| Input clip | 160 RGB frames at 128 × 128 |

Source: [TinyHR technical report, pages 6-7](website/public/downloads/tinyhr-technical-report.pdf).

## Model Size and Latency

| Measurement | Result | Interpretation |
|---|---:|---|
| ONNX parameters | **82,177** | Counted from the initializers in the published model |
| ONNX file size | **381 KiB** | SHA-256 verified against the model URL in `tiny-hr.yml` |
| TinyHR processing latency | **132 ms mean** | Input normalization + ONNX inference + waveform normalization + heart-rate signal processing |
| Median / P95 latency | **125 / 164 ms** | 50 warmed runs using ONNX Runtime CPUExecutionProvider |
| Initial observation window | **≈ 5.3 s at 30 FPS** | 160 frames must be collected before the first model estimate |
| Rolling update interval | **1.0 s default** | A new estimate is requested every 30 frames at 30 FPS |

Benchmark environment: Apple M4 MacBook Air, 10-core CPU, 24 GB memory, ONNX Runtime
1.30.0, CPUExecutionProvider, 11 September 2026. The compute benchmark excludes camera
capture and face detection, whose cost depends on the deployed detector and hardware.

The separation between observation time and compute time is important in real use. A
short facial-video window supplies enough temporal information for pulse estimation;
after that warm-up, sub-second TinyHR processing fits comfortably inside the default
one-second rolling update cadence. This makes the model useful for live wellness
interfaces, human-computer interaction, affective-computing research, and lightweight
remote monitoring prototypes.

## How TinyHR Works

[![TinyHR architecture and inference flow](website/public/media/tinyhr-flowchart.png)](website/public/downloads/tinyhr-flowchart.pdf)

*Click the diagram to open the full architecture PDF.*

| Stage | Module | Function |
|---:|---|---|
| 01 | Frame Difference Fusion Stem | Converts neighboring-frame differences into compact spatial features that emphasize subtle temporal color changes. |
| 02 | Spatial Patch Embedding | Reduces each feature map from 32 × 32 to 8 × 8 while preserving the temporal sequence. |
| 03 | Multi-scale Temporal Feature Block | Combines temporal information at several offsets with residual feature fusion. |
| 04 | Waveform Predictor | Produces one rPPG waveform value for every input frame. |
| 05 | Signal Processing | Applies detrending, band-pass filtering, and Welch spectral analysis to obtain BPM. |

During inference, the physiological band is restricted to 0.75-2.5 Hz, corresponding
to 45-150 BPM. The dominant frequency of the filtered waveform is converted to heart
rate:

```text
Heart rate (BPM) = 60 × dominant frequency (Hz)
```

Training combines waveform agreement, frequency-domain classification, and
heart-rate distribution objectives:

```text
L = 0.2 L_time + L_CE + L_KL
```

## Evaluation

| Dataset | Test split | Protocol | MAE |
|---|---|---|---:|
| VIPL-HR V1 | 22 subjects · 485 videos | Held out from training | **3.88 BPM** |

This is the author-reported result from the supplied technical report. It describes the
specified test set rather than a guaranteed error bound for every video.

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
seetapsych-webui --files seetapsych_hertz/modules/seeta.yml
```

### Programmatic Usage

```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_hertz/modules/seeta.yml")

pipeline = Pipeline(factory, ...)
pipeline.add_attributes("face/heart_rate")
```

## Module Zoo

| Module | Description | Input modes |
|---|---|---|
| [AdaChrom](seetapsych_hertz/modules/ada-chrom.yml) | Model-free adaptive chrominance rPPG on skin ROIs | Video stream · video file |
| [Seeta](seetapsych_hertz/modules/seeta.yml) | Seeta heart-rate estimation module | Video stream · video file |
| [TinyHR](seetapsych_hertz/modules/tiny-hr.yml) | Lightweight waveform model with Welch spectral analysis | Video stream · video file |

### TinyHR Parameters

| Name | Type | Default | Description |
|---|---|---:|---|
| `fps` | number | `30` | Expected video frame rate used for buffering and spectral analysis |
| `interval` | number | `1` | Seconds between consecutive rolling heart-rate estimates |

### AdaChrom Parameters

| Name | Type | Default | Description |
|---|---|---|---|
| `window_samples` | integer | `300` | Sliding-window frame count; larger values reduce noise but increase response time |
| `roi_regions` | `selection[]` | `["skin_b_adaptive_forehead"]` | Skin regions evaluated before valid estimates are fused |

Both modules provide the
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
