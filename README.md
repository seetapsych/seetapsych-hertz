# SeetaPsych Hertz

> Heart rate estimation modules for SeetaPsych

## Usage

This project is already included in the seetapsych-lib default configuration. Download and use it via `seetapsych-manager download`.

For usage, refer to [SeetaPsych](https://github.com/seetapsych/seetapsych-lib).

You can additionally add this algorithm module using the following methods.

Heart rate estimation requires processing video or real-time video streams to extract heart rate information.

### WebUI

Run `seetapsych-webui` with the `--files` argument to use it.

```
seetapsych-webui --files seetapsych_hertz/modules/seeta.yml
```

### Programmatic Usage

Add the following code in your program to use this algorithm module.
```python
from seetapsych_lib.runtime.factory import Factory
from seetapsych_lib.runtime.pipeline import Pipeline

factory = Factory()
factory.load_file_modules("seetapsych_hertz/modules/seeta.yml")

pipeline = Pipeline(factory, ...)

pipeline.add_attributes("face/heart_rate")
```

## Introduction

### SeetaHeartRateDetector (Signal-based)

Traditional signal-processing-based heart rate estimation. Extracts ROI from facial dense landmarks, then applies chrominance analysis with cubic spline interpolation and FFT frequency-domain peak detection to estimate heart rate from BGR channel signals.

Module config: [seeta.yml](seetapsych_hertz/modules/seeta.yml).
Provide Attributes: `face/heart_rate`.

Requires: `face/dense_landmarks`.

Parameters:
- `min_seconds` (float, default `1`): Minimum signal duration in seconds before estimation starts.
- `min_frames` (int, default `10`): Minimum number of frames before estimation starts.
- `max_frames` (int, default `300`): Maximum frames kept in the sliding signal window.

### AdaChrom (Adaptive Chrominance)

Adaptive chrominance-based heart rate estimation using frame-wise sliding window analysis. Extracts photoplethysmographic signals from an adaptive forehead skin ROI, processes each frame with chrominance methods to produce real-time BPM output without FFT fusion.

Module config: [ada-chrom.yml](seetapsych_hertz/modules/ada-chrom.yml).
Provide Attributes: `face/heart_rate`.

Requires: `face/dense_landmarks`.

Parameters:
- `window_samples` (int, default `300`): Number of frames in the sliding estimation window.

### TinyHR (ONNX-based)

RhythmFormer ONNX heart rate estimation engine. Converts face-frame sequences into BVP signals using a deep neural network, then applies long-window BVP fusion and Welch spectral analysis for stable heart-rate output.

Module config: [tiny-hr.yml](seetapsych_hertz/modules/tiny-hr.yml).
Provide Attributes: `face/heart_rate`.

Requires: `face/detection`.

Available model: `seeta-hertz-tinyhr.onnx`.

Parameters:
- `fps` (float, default `30`): Expected camera/video FPS.
- `interval` (float, default `1`): Interval in seconds between heart rate estimations.
