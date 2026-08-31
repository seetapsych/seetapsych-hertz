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

### Module Catalog

| Module YAML Path | Package Name |
|---|---|
| `seetapsych_hertz/modules/ada-chrom.yml` | HeartRate-AdaChrom |
| `seetapsych_hertz/modules/seeta.yml` | HeartRate-Seeta |
| `seetapsych_hertz/modules/tiny-hr.yml` | HeartRate-TinyHR |

### AdaChrom

> Model-free rPPG heart rate estimation using adaptive chrominance analysis on skin ROI.

Module config: [ada-chrom.yml](seetapsych_hertz/modules/ada-chrom.yml)

| Package | Provides | Requires |
|---|---|---|
| HeartRate-AdaChrom | `face/heart_rate` | `face/dense_landmarks` |

**Description**

Adaptive chrominance rPPG heart rate estimator from adaptive forehead ROI, no neural model required.

**Parameters**

| Name | Type | Default | Description & Tuning |
|---|---|---|---|
| `window_samples` | integer | `300` | Sliding window frame count for HR estimation. Larger values reduce noise but increase latency; adjust based on real-time demand. |

**Models**

*(None)*

### SeetaHeartRateDetector

> Traditional signal-processing heart rate estimation from forehead skin pixels via chrominance + FFT analysis.

Module config: [seeta.yml](seetapsych_hertz/modules/seeta.yml)

| Package | Provides | Requires |
|---|---|---|
| HeartRate-Seeta | `face/heart_rate` | `face/dense_landmarks` |

**Description**

Signal-processing heart rate estimator using forehead ROI from dense landmarks with chrominance + FFT peak detection.

**Parameters**

| Name | Type | Default | Description & Tuning |
|---|---|---|---|
| `min_seconds` | number | `1` | Minimum buffered signal duration (seconds) before producing first HR estimate. Lower for faster startup, higher for stability. |
| `min_frames` | integer | `10` | Minimum frame count before HR estimation starts. Takes effect whichever is reached later between min_seconds and min_frames. |
| `max_frames` | integer | `300` | Maximum frames kept in the sliding analysis window. Larger windows smooth outliers but increase response lag to true HR changes. |

**Models**

*(None)*

### TinyHR

> Lightweight neural network for fast heart rate estimation directly from face video frames.

Module config: [tiny-hr.yml](seetapsych_hertz/modules/tiny-hr.yml)

| Package | Provides | Requires |
|---|---|---|
| HeartRate-TinyHR | `face/heart_rate` | `face/detection` |

**Description**

Fast RhythmFormer heart rate estimator using buffered face crops + Welch spectral analysis.

**Parameters**

| Name | Type | Default | Description & Tuning |
|---|---|---|---|
| `fps` | number | `30` | Expected camera/video FPS used for spectral analysis windowing. Mismatch with actual source FPS degrades HR accuracy. |
| `interval` | number | `1` | Seconds between consecutive HR estimates. Smaller intervals yield more updates with higher jitter; larger intervals are smoother but slower. |

**Models**

| Model | Recommended |
|---|---|
| `seeta-hertz-tinyhr.onnx` | ✓ |
