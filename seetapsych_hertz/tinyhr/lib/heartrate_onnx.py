#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RhythmFormer ONNX Heart Rate Estimation SDK

Features:
1. BVPONNX: loads the ONNX model and converts a face-frame sequence into a normalized BVP signal.
2. Long-window BVP fusion: merges multiple short BVP segments into a 20-second BVP signal to improve heart-rate stability.
3. Streaming camera heart-rate estimation: accepts externally supplied one-second frame batches, maintains an internal frame buffer, and returns heart rate.

Three usage modes:
    1) Video file mode: estimate_video_hr(video_path, fps) → dict
    2) NumPy array mode: estimate_array_hr(frames_5d, fps) → dict
    3) Streaming camera mode: CameraHRTracker → push(second_frames) → bpm

Dependencies:
    onnxruntime, numpy, scipy, opencv-python, pillow
"""

import os
import time
import cv2
import numpy
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, welch
from scipy.sparse import spdiags
import onnxruntime as ort


# ============================================================
#  Default configuration
# ============================================================

# --- Model paths ---
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "heartrate.onnx")

# --- Model input configuration ---
INPUT_FRAMES = 160       # Number of model input frames
INPUT_CHANNELS = 3       # RGB channels
INPUT_SIZE = 128         # Face ROI size (128x128)

# --- Face detection settings ---
FACE_BOX_SCALE = 1.5     # Face bounding-box scale factor
FACE_DET_INTERVAL = 1    # Face detection interval (seconds)

# --- Heart-rate calculation parameters ---
HR_MIN_BPM = 45          # Minimum heart rate (bpm)
HR_MAX_BPM = 150         # Maximum heart rate (bpm)
FILTER_LOW_FREQ = 0.75   # Band-pass low cutoff (Hz) = 45 bpm
FILTER_HIGH_FREQ = 2.5   # Band-pass high cutoff (Hz) = 150 bpm
SAMPLE_RATE = 30         # Video sampling rate (Hz)
NPERSEG = 256            # Window length for the Welch method

# --- Long BVP window configuration ---
LONG_BVP_DURATION = 20.0 # Long BVP window duration (seconds)

# --- Streaming parameters ---
FPS_DEFAULT = 30         # Default video frame rate

# --- Debug output ---
VERBOSE = False


# ============================================================
#  Helper functions: signal processing
# ============================================================

def _detrend(input_signal, lambda_value=100):
    """
    Remove the trend component from a PPG/BVP signal.

    Args:
        input_signal: Input signal array.
        lambda_value: Smoothing parameter.

    Returns:
        detrended_signal: Detrended signal.
    """
    signal_length = input_signal.shape[0]
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal


def bandpass_filter(signal, fs=SAMPLE_RATE, low=FILTER_LOW_FREQ, high=FILTER_HIGH_FREQ):
    """
    Apply a first-order Butterworth band-pass filter.

    Args:
        signal: Input signal.
        fs: Sampling rate (Hz).
        low: Low-frequency cutoff (Hz).
        high: High-frequency cutoff (Hz).

    Returns:
        filtered: Filtered signal.
    """
    nyquist = fs / 2
    b, a = butter(1, [low / nyquist, high / nyquist], btype="bandpass")
    return filtfilt(b, a, signal.astype(np.float64))


def calculate_hr(bvp_signal, fs=SAMPLE_RATE):
    """
    Estimate heart rate by applying Welch spectral analysis to the BVP signal.

    Args:
        bvp_signal: BVP signal array.
        fs: Sampling rate (Hz).

    Returns:
        hr: Heart rate in bpm, or None if it cannot be calculated.
    """
    freqs, psd = welch(
        bvp_signal,
        fs=fs,
        nfft=int(1e5 / fs),
        nperseg=min(len(bvp_signal) - 1, NPERSEG),
    )

    valid_mask = (freqs > HR_MIN_BPM / 60) & (freqs < HR_MAX_BPM / 60)
    valid_freqs = freqs[valid_mask]
    valid_psd = psd[valid_mask]

    if len(valid_psd) == 0:
        return None

    bpm_freq = valid_freqs[np.argmax(valid_psd)]
    hr = bpm_freq * 60
    return hr


# ============================================================
#  Helper functions: face bounding-box operations
# ============================================================

def expand_box(box, scale=FACE_BOX_SCALE, img_w=0, img_h=0):
    """
    Expand a bounding box by the specified scale while keeping it within image boundaries.

    Args:
        box: [x1, y1, x2, y2].
        scale: Expansion scale factor.
        img_w: Image width (optional).
        img_h: Image height (optional).

    Returns:
        expanded_box: [x1, y1, x2, y2].
    """
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    w = (box[2] - box[0]) * scale
    h = (box[3] - box[1]) * scale

    x1 = max(0, int(cx - w / 2))
    y1 = max(0, int(cy - h / 2))
    x2 = min(img_w if img_w > 0 else 99999, int(cx + w / 2))
    y2 = min(img_h if img_h > 0 else 99999, int(cy + h / 2))

    return [x1, y1, x2, y2]


def compute_iou(box1, box2):
    """
    Calculate the IoU (Intersection over Union) of two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2].
        box2: [x1, y1, x2, y2].

    Returns:
        iou: IoU value in [0, 1].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ============================================================
#  Long-window BVP fusion buffer (overwrite mode without windowing)
# ============================================================

class LongBVPBuffer:
    """
    Long-window BVP fusion buffer.

    Updated strategy:
    - Each newly predicted BVP segment directly overwrites the overlapping
      INPUT_FRAMES - STEP_SIZE values in the existing buffer.
    - Every position has a weight of 1; no Hanning window is applied.
    - A longer temporal window is used to gradually stabilize each heart-rate estimate.

    Workflow:
    1. Normalize each short BVP segment.
    2. Directly overwrite the corresponding region in the buffer with unit weight.
    3. Extract the most recent LONG_BVP_DURATION seconds as the long BVP signal.
    4. Detrend and band-pass filter the long BVP, then estimate heart rate in the frequency domain.
    """

    def __init__(self, duration=LONG_BVP_DURATION, max_frames=INPUT_FRAMES, step_frames=30):
        """
        Args:
            duration: Long BVP window duration in seconds; default is 20.
            max_frames: Number of model input frames; default is 160.
            step_frames: Number of frames advanced per update (about 1 second); default is 30.
        """
        self.duration = duration
        self.window_size = max_frames
        self.step_size = step_frames

        # Accumulation buffer (all weights are fixed at 1)
        self._bvp_buffer = None
        self._bvp_fs = None
        self._global_idx = 0
        self._initialized = False

        # Current long BVP signal
        self.long_bvp = None

    def _init_buffer(self, fs):
        """Initialize the accumulation buffer"""
        max_len = int(self.duration * fs) + self.window_size
        self._bvp_buffer = np.zeros(max_len, dtype=np.float32)
        self._bvp_fs = fs
        self._global_idx = 0
        self._initialized = True

    def update(self, bvp_segment, fps):
        """
        Update the long BVP buffer using overwrite mode without windowing.

        Args:
            bvp_segment: Normalized BVP segment for the current window,
                         shape: (INPUT_FRAMES,).
            fps: Frame rate.

        Returns:
            long_bvp: Updated long BVP signal, shape: (N,).
            hr: Heart rate estimated from the long BVP in bpm, or None if unavailable.
        """
        seg = bvp_segment.astype(np.float32)
        T = len(seg)

        # Initialize the buffer
        if not self._initialized:
            self._init_buffer(fps)

        fs = self._bvp_fs
        max_len = len(self._bvp_buffer)

        # Calculate the current segment position in the buffer
        start_idx = max(0, self._global_idx - T + self.step_size)
        end_idx = start_idx + T

        # Shift the buffer left when the segment exceeds its end
        if end_idx > max_len:
            shift = end_idx - max_len
            self._bvp_buffer[:-shift] = self._bvp_buffer[shift:]
            self._bvp_buffer[-shift:] = 0.0
            start_idx -= shift
            end_idx -= shift

        # Normalize the segment
        seg_norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-7)
        seg_norm[np.isnan(seg_norm)] = 0.0

        # Direct overwrite without windowing; weight is always 1
        self._bvp_buffer[start_idx:end_idx] = seg_norm

        # Advance the global index
        if self._global_idx == 0:
            self._global_idx = T
        else:
            self._global_idx += self.step_size
        self._global_idx = min(self._global_idx, max_len)

        # Extract the long BVP from the most recent duration seconds
        keep_len = int(self.duration * fs)
        cur_end = self._global_idx
        cur_start = max(0, cur_end - keep_len)

        long_bvp = self._bvp_buffer[cur_start:cur_end].copy()

        # Standardize the complete long BVP signal
        if long_bvp.std() > 1e-7:
            long_bvp = (long_bvp - long_bvp.mean()) / (long_bvp.std() + 1e-7)

        # Detrend and apply band-pass filtering
        long_bvp = _detrend(long_bvp, lambda_value=100)
        long_bvp = bandpass_filter(long_bvp, fs=fs)

        self.long_bvp = long_bvp

        # Estimate heart rate from the long BVP
        hr = calculate_hr(long_bvp, fs=fs)

        return long_bvp, hr

    def reset(self):
        """Reset the buffer"""
        self._bvp_buffer = None
        self._bvp_fs = None
        self._global_idx = 0
        self._initialized = False
        self.long_bvp = None


# ============================================================
#  Core class: ONNX BVP inference
# ============================================================

class BVPONNX:
    """
    RhythmFormer ONNX BVP inference engine.

    Responsibilities:
    - Load and manage the ONNX model.
    - Apply Z-score preprocessing.
    - Run inference on CPU or GPU.
    - Return a normalized BVP signal without post-processing.

    Input: RGB face video clip with shape
           (INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE, 3).
    Output: Normalized BVP signal with shape (INPUT_FRAMES,).

    Input format:
        The ONNX model expects NCHW-like video input:
        (1, 160, 3, 128, 128), where batch=1, frames=160,
        channels=3, height=128, and width=128.
    """

    def __init__(self, onnx_model_path=None, model_dir=None, providers=None, verbose=VERBOSE):
        """
        Initialize the ONNX BVP inference engine.

        Args:
            onnx_model_path: Path to the ONNX model file. If None, search automatically.
            model_dir: Directory containing model files.
            providers: ONNX Runtime execution provider list. If None, try CUDA and then CPU.
            verbose: Whether to print detailed information.
        """
        self.verbose = verbose
        self.session = None

        # Configure inference execution providers
        if providers is None:
            # Try CUDA first and fall back to CPU
            try:
                providers = [("CUDAExecutionProvider"), ("CPUExecutionProvider")]
            except Exception:
                providers = ["CPUExecutionProvider"]

        # Resolve the model path
        if onnx_model_path is None:
            if model_dir is None:
                model_dir = DEFAULT_MODEL_DIR
            onnx_model_path = self._find_model(model_dir, "heartrate")

        if not os.path.exists(onnx_model_path):
            raise FileNotFoundError(f"ONNX model file does not exist: {onnx_model_path}")

        if self.verbose:
            print(f"[INFO] Loading RhythmFormer ONNX model: {onnx_model_path}")

        # Load the ONNX model
        self.session = ort.InferenceSession(
            onnx_model_path,
            providers=providers
        )

        # Get model input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        if self.verbose:
            print("[INFO] ONNX model loaded")

            # Print model information
            input_shape = self.session.get_inputs()[0].shape
            output_shape = self.session.get_outputs()[0].shape
            print(f"[INFO] Model input name: {self.input_name}, shape: {input_shape}")
            print(f"[INFO] Model output name: {self.output_name}, shape: {output_shape}")

        # Model input configuration
        self.input_frames = INPUT_FRAMES
        self.input_channels = INPUT_CHANNELS
        self.input_size = INPUT_SIZE

        # Verify model I/O
        self._verify_model_io()

    def _find_model(self, model_dir, keyword="heartrate"):
        """Find a .onnx model file containing keyword in the directory"""
        if not os.path.exists(model_dir):
            return DEFAULT_MODEL_PATH

        model_files = [f for f in os.listdir(model_dir)
                       if f.endswith('.onnx') and keyword in f]
        if not model_files:
            # Fallback: find any .onnx file
            model_files = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]

        if not model_files:
            return DEFAULT_MODEL_PATH

        model_files.sort()
        return os.path.join(model_dir, model_files[-1])

    def _verify_model_io(self):
        """Verify model input/output shapes."""
        if self.verbose:
            print("[INFO] Verifying model input/output...")

        test_input = np.random.randn(
            1, self.input_frames, self.input_channels,
            self.input_size, self.input_size
        ).astype(np.float32)

        t1 = time.time()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: test_input}
        )
        t2 = time.time()

        if outputs is None or len(outputs) == 0:
            raise RuntimeError("ONNX inference returned an empty result")

        output = outputs[0]
        output_shape = output.shape
        expected_shape = (1, self.input_frames)

        if output_shape != expected_shape:
            print(f"[WARN] Model output shape: {output_shape}, expected shape: {expected_shape}")

        if self.verbose:
            print(f"[INFO] Model input shape: (1, {self.input_frames}, {self.input_channels}, "
                  f"{self.input_size}, {self.input_size})")
            print(f"[INFO] Model output shape: {output_shape}")
            print(f"[INFO] Inference time: {(t2-t1)*1000:.1f} ms")

    def preprocess(self, video_faces):
        """
        Preprocess input using Z-score normalization and dimension reordering.

        Args:
            video_faces: NumPy array with shape
                         (INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE, 3),
                         RGB format, value range [0, 255].

        Returns:
            standardized: float32 array with shape
                          (1, INPUT_FRAMES, 3, INPUT_SIZE, INPUT_SIZE),
                          matching the ONNX model input layout.
        """
        frames = video_faces.astype(np.float32)

        mean = np.mean(frames)
        std = np.std(frames)
        if std < 1e-7:
            std = 1e-7

        standardized = (frames - mean) / std
        standardized[np.isnan(standardized)] = 0.0

        # (T, H, W, C) -> (T, C, H, W) -> (1, T, C, H, W)
        # ONNX input layout: NCHW
        standardized = np.transpose(standardized, (0, 3, 1, 2))
        standardized = np.expand_dims(standardized, axis=0)

        return standardized

    def infer(self, video_faces):
        """
        Run one inference pass from face frames to a normalized BVP signal.

        This method only performs model inference and BVP normalization.
        It does not perform detrending, filtering, or heart-rate calculation;
        those post-processing steps are handled by the caller.

        Args:
            video_faces: RGB face frames with shape
                         (INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE, 3).

        Returns:
            normalized_bvp: Normalized BVP signal with shape (INPUT_FRAMES,),
                            or None if inference fails.
        """
        # Preprocess
        input_data = self.preprocess(video_faces)

        # ONNX inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )

        if outputs is None or len(outputs) == 0:
            return None

        # Extract and normalize the BVP signal
        bvp_raw = outputs[0].flatten()  # (INPUT_FRAMES,)
        normalized_bvp = (bvp_raw - np.mean(bvp_raw)) / (np.std(bvp_raw) + 1e-7)
        normalized_bvp[np.isnan(normalized_bvp)] = 0.0

        return normalized_bvp

    def cleanup(self):
        """Release ONNX resources"""
        if self.session is not None:
            self.session = None
            if self.verbose:
                print("[INFO] ONNX resources released")

    def __del__(self):
        self.cleanup()


# ============================================================
#  Streaming camera heart-rate estimator
# ============================================================

class CameraHRTracker:
    """
    Streaming camera heart-rate estimator

    Design:
    - Does not open the camera itself; the caller supplies about one second of raw video frames.
    - FPS is specified during initialization.
    - Maintains an internal frame queue and estimates heart rate only when its length is at least INPUT_FRAMES.
    - Uses long-window BVP fusion to gradually stabilize heart-rate estimates.
    - Returns None when the queue is not full enough or no face is available.

    Example:
        tracker = CameraHRTracker(fps=30, model_path="...")
        tracker.start_face_detection()  # Start the face detector

        while True:
            # Supply about one second of frames: shape (N, H, W, 3), BGR
            second_frames = get_next_second_of_frames()
            bpm = tracker.push(second_frames)
            if bpm is not None:
                print(f"Current heart rate: {bpm:.1f} bpm")
            else:
                print("Waiting for enough frames and a detected face...")

        tracker.cleanup()
    """

    def __init__(self, fps=FPS_DEFAULT, model_path=None, model_dir=None,
                 providers=None, verbose=VERBOSE):
        """
        Args:
            fps: Video frame rate used for the long BVP buffer and face-detection interval.
            model_path: ONNX Model paths
            model_dir: Model directory.
            providers: ONNX Runtime execution provider list.
            verbose: Whether to print detailed information.
        """
        self.fps = fps
        self.verbose = verbose
        self.step_frames = fps  # Number of frames per second

        # Initialize the ONNX BVP inference engine
        self.bvp_estimator = BVPONNX(
            onnx_model_path=model_path, model_dir=model_dir,
            providers=providers, verbose=verbose
        )

        # Initialize the long-window BVP fusion buffer
        self.long_bvp_buffer = LongBVPBuffer(
            duration=LONG_BVP_DURATION,
            max_frames=INPUT_FRAMES,
            step_frames=self.step_frames
        )

        # Frame queue for maintaining the fixed-size temporal window
        self.frame_queue = deque(maxlen=INPUT_FRAMES + self.step_frames)

        # Face detection state
        self.face_box = None
        self.has_face = False
        self.last_det_frame_idx = -1

        # Face detector (initialized lazily)
        self._face_detector_initialized = False
        self._face_detection_failed = False

        # Runtime state
        self._total_frames = 0

        if self.verbose:
            print(f"[INFO] CameraHRTracker initialized")
            print(f"  - Frame rate: {fps} FPS")
            print(f"  - Frame queue capacity: {INPUT_FRAMES + self.step_frames}")
            print(f"  - Long BVP window: {LONG_BVP_DURATION} seconds")

    def start_face_detection(self, face_model_path=None):
        """
        Start the face detector using face_detection_onnx.FaceDetector_ONNX.

        Args:
            face_model_path: Path to the ONNX face-detection model; if None, use the default model path.
        """
        if self._face_detector_initialized:
            return

        try:
            from face_detection_onnx import FaceDetector_ONNX, FACE_DETECTOR_MODEL_ONNX
            if face_model_path is None:
                face_model_path = FACE_DETECTOR_MODEL_ONNX
            self._face_detector = FaceDetector_ONNX(onnx_model_path=face_model_path)
            self._face_detector_initialized = True
            if self.verbose:
                print("[INFO] Face detector started (FaceDetector_ONNX)")
        except ImportError:
            self._face_detection_failed = True
            if self.verbose:
                print("[WARN] Failed to import face_detection_onnx; face detection is unavailable")
        except Exception as e:
            self._face_detection_failed = True
            if self.verbose:
                print(f"[WARN] Failed to start face detector: {e}")

    def _detect_face(self, frame_bgr):
        """
        Detect a face using FaceDetector_ONNX.

        Args:
            frame_bgr: Frame in BGR format.

        Returns:
            box: [x1, y1, x2, y2], or None if no face is detected.
        """
        if not self._face_detector_initialized or self._face_detection_failed:
            return None

        try:
            bboxes, lms, _ = self._face_detector.forward(frame_bgr)
            if len(bboxes) == 0:
                return None
            # Select the face bounding box with the largest area
            bboxes_np = np.array(bboxes)
            areas = (bboxes_np[:, 2] - bboxes_np[:, 0]) * (bboxes_np[:, 3] - bboxes_np[:, 1])
            best_idx = np.argmax(areas)
            return bboxes_np[best_idx][:4].tolist()
        except Exception as e:
            if self.verbose:
                print(f"[WARN] Face detection failed: {e}")
            self._face_detection_failed = True
            return None

    def push(self, frames: numpy.ndarray, faces: numpy.ndarray = None):
        """
        Push approximately one second of video frames.

        Args:
            frames: NumPy array with shape (N, H, W, 3), using BGR channel order;
                    N is typically equal to fps.
            faces: Optional NumPy array with shape (N, 4), where each row is
                   [x1, y1, x2, y2].

        Returns:
            bpm: Heart rate in bpm, or None if the queue is not full enough
                 or no face is available.
        """
        if not isinstance(frames, np.ndarray) or len(frames.shape) != 4:
            if self.verbose:
                print("[WARN] Invalid frame data format; expected (N, H, W, 3)")
            return None

        if faces is not None:
            if not isinstance(faces, np.ndarray) or len(faces.shape) != 2:
                if self.verbose:
                    print("[WARN] Invalid face data format; expected (N, 4)")
                return None

        n_frames = frames.shape[0]
        if n_frames == 0:
            return None

        # Update the global frame counter
        self._total_frames += n_frames

        # --- Face detection (once per second)---
        if self.face_box is None or (self._total_frames - self.last_det_frame_idx) >= self.step_frames:
            # Use the BGR frame for face detection
            first_frame_bgr = frames[0]
            if faces is None:
                new_box = self._detect_face(first_frame_bgr)
            else:
                new_box = faces[0].tolist()
            if new_box is not None:
                if self.has_face:
                    # Use IoU to decide whether to update the box; a low IoU may indicate a different face
                    iou = compute_iou(self.face_box, new_box)
                    if iou < 0.8:
                        self.face_box = new_box
                else:
                    self.face_box = new_box
                    self.has_face = True
                self.last_det_frame_idx = self._total_frames
            elif self.has_face:
                # A face was previously present but is now missing; clear the queue
                self.has_face = False
                self.frame_queue.clear()

        # --- Process frames and enqueue cropped faces ---
        img_h, img_w = frames.shape[1], frames.shape[2]

        if self.has_face and self.face_box is not None:
            expanded_box = expand_box(
                self.face_box,
                scale=FACE_BOX_SCALE,
                img_w=img_w,
                img_h=img_h
            )

            for i in range(n_frames):
                face = self._crop_face(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), expanded_box)
                if face is not None:
                    self.frame_queue.append(face)
        # else: do not enqueue frames when no face is present; the queue was cleared when the face was lost

        # --- Estimate heart rate once the queue contains at least INPUT_FRAMES frames ---
        if len(self.frame_queue) >= INPUT_FRAMES:
            cropped = np.array(list(self.frame_queue)[-INPUT_FRAMES:])

            # BVPONNX returns only the normalized BVP signal
            normalized_bvp = self.bvp_estimator.infer(cropped)

            if normalized_bvp is not None:
                # Update the long-window BVP fusion buffer
                long_bvp, long_hr = self.long_bvp_buffer.update(normalized_bvp, self.fps)

                return long_hr

        return None

    def get_long_bvp(self):
        """Return the current long BVP signal."""
        return self.long_bvp_buffer.long_bvp

    def get_buffer_info(self):
        """Return buffer status"""
        return {
            "frame_queue_length": len(self.frame_queue),
            "has_face": self.has_face,
            "total_frames": self._total_frames,
            "long_bvp_length": len(self.long_bvp_buffer.long_bvp)
            if self.long_bvp_buffer.long_bvp is not None else 0,
        }

    def reset(self):
        """Reset all state"""
        self.frame_queue.clear()
        self.long_bvp_buffer.reset()
        self.face_box = None
        self.has_face = False
        self.last_det_frame_idx = -1
        self._total_frames = 0

    def _crop_face(self, frame_rgb, box):
        """Crop and resize the face"""
        x1, y1, x2, y2 = [int(b) for b in box]
        h, w = frame_rgb.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_rgb[y1:y2, x1:x2].copy()
        size = self.bvp_estimator.input_size
        face = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
        return face

    def cleanup(self):
        """Release resources"""
        self.bvp_estimator.cleanup()

    def __del__(self):
        self.cleanup()


# ============================================================
#  Usage mode 1: video file
# ============================================================

def estimate_video_hr(video_path, fps=FPS_DEFAULT, model_path=None, model_dir=None,
                      verbose=VERBOSE):
    """
    Estimate heart rate from a video file.

    Workflow:
    1. Open the video file and accumulate at least INPUT_FRAMES frames.
    2. Detect a face, expand the bounding box by 1.5x, and crop the face region.
    3. Detect the face once per second thereafter; skip heart-rate estimation if the face is lost.
    4. Use long-window BVP fusion to calculate heart rate.

    Args:
        video_path: Path to the video file.
        fps: VideoFrame rate
        model_path: Optional ONNX model path.
        model_dir: Optional model directory.
        verbose: Whether to print detailed information.

    Returns:
        result: Dictionary containing:
            - heart_rates: List of (frame_idx, bpm) heart-rate records.
            - avg_hr: Average heart rate (bpm)
            - min_hr: Minimum heart rate
            - max_hr: Maximum heart rate
            - std_hr: Heart-rate standard deviation
    """
    estimator = VideoFileHR(
        video_path=video_path, fps=fps,
        model_path=model_path, model_dir=model_dir,
        verbose=verbose
    )
    return estimator.run()


class VideoFileHR:
    """
    Video-file heart-rate estimator.

    Example:
        result = estimate_video_hr("video.mp4", fps=30)
        print(f"Average heart rate: {result['avg_hr']:.1f} bpm")
    """

    def __init__(self, video_path, fps=FPS_DEFAULT, model_path=None, model_dir=None,
                 verbose=VERBOSE):
        self.video_path = video_path
        self.fps = fps
        self.verbose = verbose

        # Initialize the ONNX BVP inference engine
        self.bvp_estimator = BVPONNX(
            onnx_model_path=model_path, model_dir=model_dir, verbose=verbose
        )

        # Initialize the long-window BVP fusion buffer
        self.long_bvp_buffer = LongBVPBuffer(
            duration=LONG_BVP_DURATION,
            max_frames=INPUT_FRAMES,
            step_frames=int(fps)  # Advance once per second
        )

        # Open the video
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        # Video metadata
        self.actual_fps = fps
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0

        # Face detection state
        self.face_box = None
        self.has_face = False
        self.last_det_frame = -1

        # Face detector
        self._face_detector_initialized = False
        self._face_detection_failed = False
        self._init_face_detector()

        # Frame buffer
        self.frame_buffer = []

        # Heart-rate calculation state
        self.frames_since_last_calc = 0  # Number of frames since the previous heart-rate calculation

        # Results
        self.heart_rates = []

        if self.verbose:
            print(f"[INFO] Video: {video_path}")
            print(f"[INFO] FPS: {self.actual_fps:.2f}, Total frames: {self.total_frames}")

    def _init_face_detector(self, face_model_path=None):
        """Initialize the FaceDetector_ONNX face detector."""
        try:
            from face_detection_onnx import FaceDetector_ONNX, FACE_DETECTOR_MODEL_ONNX
            if face_model_path is None:
                face_model_path = FACE_DETECTOR_MODEL_ONNX
            self._face_detector = FaceDetector_ONNX(onnx_model_path=face_model_path)
            self._face_detector_initialized = True
        except Exception:
            self._face_detection_failed = True

    def _detect_face(self, frame_bgr):
        """Detect a face using FaceDetector_ONNX"""
        if not self._face_detector_initialized or self._face_detection_failed:
            return None
        try:
            bboxes, lms, _ = self._face_detector.forward(frame_bgr)
            if len(bboxes) == 0:
                return None
            bboxes_np = np.array(bboxes)
            areas = (bboxes_np[:, 2] - bboxes_np[:, 0]) * (bboxes_np[:, 3] - bboxes_np[:, 1])
            best_idx = np.argmax(areas)
            return bboxes_np[best_idx][:4].tolist()
        except Exception:
            self._face_detection_failed = True
            return None

    def _crop_face(self, frame_rgb, box):
        """Crop and resize according to the face bounding box"""
        x1, y1, x2, y2 = [int(b) for b in box]
        h, w = frame_rgb.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame_rgb[y1:y2, x1:x2].copy()
        face = cv2.resize(crop, (self.bvp_estimator.input_size, self.bvp_estimator.input_size),
                          interpolation=cv2.INTER_AREA)
        return face

    def run(self):
        """Run heart-rate estimation on the video file"""
        if self.verbose:
            print("\n[INFO] Starting video processing...\n")

        t_start = time.time()

        while True:
            ret, frame_bgr = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # --- Face detection on the first frame and once per second thereafter---
            frames_since_det = self.frame_idx - self.last_det_frame
            fps_int = int(round(self.actual_fps))

            if self.face_box is None or frames_since_det >= fps_int:
                new_box = self._detect_face(frame_bgr)
                # print(f'new_box: {new_box}')
                if new_box is not None:
                    if self.has_face:
                        iou = compute_iou(self.face_box, new_box)
                        if iou < 0.8:
                            self.face_box = new_box
                    else:
                        self.face_box = new_box
                        self.has_face = True
                    self.last_det_frame = self.frame_idx
                elif self.has_face:
                    # Face lost
                    self.has_face = False
                    self.frame_buffer.clear()

            # --- Accumulate frames and crop the face ---
            if self.has_face and self.face_box is not None:
                expanded_box = expand_box(
                    self.face_box,
                    scale=FACE_BOX_SCALE,
                    img_w=frame_rgb.shape[1],
                    img_h=frame_rgb.shape[0]
                )
                face = self._crop_face(frame_rgb, expanded_box)
                if face is not None:
                    self.frame_buffer.append(face)

            # --- Estimate heart rate once per second ---
            fps_int = int(round(self.actual_fps))
            if len(self.frame_buffer) >= INPUT_FRAMES and self.frames_since_last_calc >= fps_int:
                cropped = np.array(self.frame_buffer[-INPUT_FRAMES:])

                # BVPONNX returns only the normalized BVP signal
                normalized_bvp = self.bvp_estimator.infer(cropped)

                if normalized_bvp is not None:
                    # Update the long BVP signal
                    long_bvp, long_hr = self.long_bvp_buffer.update(normalized_bvp, self.fps)

                    if long_hr is not None:
                        self.heart_rates.append((self.frame_idx, long_hr))
                        if self.verbose:
                            print(f"  Frame {self.frame_idx}: BPM = {long_hr:.1f} "
                                  f"(long BVP length: {len(long_bvp)} frames)")

                # Reset the counter
                self.frames_since_last_calc = 0

            self.frames_since_last_calc += 1
            self.frame_idx += 1

        self.cap.release()
        elapsed = time.time() - t_start

        # Summarize results
        if self.heart_rates:
            bpm_values = [hr for _, hr in self.heart_rates]
            result = {
                "heart_rates": self.heart_rates,
                "avg_hr": float(np.mean(bpm_values)),
                "min_hr": float(np.min(bpm_values)),
                "max_hr": float(np.max(bpm_values)),
                "std_hr": float(np.std(bpm_values)),
                "total_frames": self.frame_idx,
                "processing_time": elapsed,
                "fps": self.frame_idx / elapsed if elapsed > 0 else 0,
            }
        else:
            result = {
                "heart_rates": [],
                "avg_hr": None,
                "min_hr": None,
                "max_hr": None,
                "std_hr": None,
                "total_frames": self.frame_idx,
                "processing_time": elapsed,
                "fps": 0,
            }

        if self.verbose:
            print(f"\n[INFO] Processing complete!")
            print(f"  Total frames: {self.frame_idx}")
            print(f"  Heart-rate estimates: {len(self.heart_rates)}")
            print(f"  Processing time: {elapsed:.2f}s")
            if self.heart_rates:
                print(f"  Average heart rate: {result['avg_hr']:.1f} bpm")
                print(f"  Minimum heart rate: {result['min_hr']:.1f} bpm")
                print(f"  Maximum heart rate: {result['max_hr']:.1f} bpm")
                print(f"  Heart-rate standard deviation: {result['std_hr']:.1f} bpm")

        self.bvp_estimator.cleanup()
        return result


# ============================================================
#  Usage mode 2: NumPy array
# ============================================================

def estimate_array_hr(frames, fps, model_path=None, model_dir=None,
                      verbose=VERBOSE):
    """
    Estimate heart rate from a NumPy array.

    The input shape is (INPUT_FRAMES, H, W, 3) with BGR channel order.
    The input is treated as a single video clip: detect a face on the first frame,
    expand the face box by 1.5x, crop the face region from every frame, and run
    model inference on the cropped face sequence.

    Args:
        frames: NumPy array with shape (INPUT_FRAMES, H, W, 3),
                   BGR format, value range [0, 255].
        fps: Video frame rate used for BVP heart-rate estimation.
        model_path: Optional ONNX model path.
        model_dir: Optional model directory.
        verbose: Whether to print detailed information.

    Returns:
        result: Dictionary containing:
            - normalized_bvp: Normalized BVP signal, shape: (INPUT_FRAMES,).
            - bpm: Heart rate in bpm, or None if it cannot be calculated.
    """
    # Validate the input shape
    original_shape = frames.shape
    if len(original_shape) == 4:
        frames_4d = frames
    else:
        raise ValueError(f"Input array must be 4-D; actual dimensions: {len(original_shape)}")

    # Validate the frame count
    if frames_4d.shape[0] != INPUT_FRAMES:
        raise ValueError(
            f"Expected frame count is {INPUT_FRAMES}，actual {frames_4d.shape[0]}"
        )

    # Initialize the face detector
    face_detector = None
    try:
        from face_detection_onnx import FaceDetector_ONNX, FACE_DETECTOR_MODEL_ONNX
        face_detector = FaceDetector_ONNX(onnx_model_path=FACE_DETECTOR_MODEL_ONNX)
    except Exception as e:
        if verbose:
            print(f"[WARN] Failed to initialize face detector: {e}")

    # Detect a face on the first frame
    first_frame_bgr = frames_4d[0].astype(np.uint8)  # (H, W, 3)
    face_box = None

    if face_detector is not None:
        try:
            bboxes, lms, _ = face_detector.forward(first_frame_bgr)
            if len(bboxes) > 0:
                # Select the face bounding box with the largest area
                bboxes_np = np.array(bboxes)
                areas = (bboxes_np[:, 2] - bboxes_np[:, 0]) * (bboxes_np[:, 3] - bboxes_np[:, 1])
                best_idx = np.argmax(areas)
                face_box = bboxes_np[best_idx][:4].tolist()
        except Exception as e:
            if verbose:
                print(f"[WARN] Face detection failed: {e}")

    if face_box is None:
        if verbose:
            print("[WARN] No face detected; heart-rate estimation cannot be performed")
        return {"normalized_bvp": None, "bpm": None}

    if verbose:
        print(f"[INFO] Detected face bounding box: {face_box}")

    # Expand the face bounding box by 1.5x
    img_h, img_w = frames_4d.shape[1], frames_4d.shape[2]
    expanded_box = expand_box(face_box, scale=FACE_BOX_SCALE, img_w=img_w, img_h=img_h)
    if verbose:
        print(f"[INFO] Expanded face bounding box: {expanded_box}")

    # Crop and resize each frame
    cropped_faces = []
    for i in range(INPUT_FRAMES):
        frame_rgb = cv2.cvtColor(frames_4d[i].astype(np.uint8), cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = [int(b) for b in expanded_box]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)

        if x2 <= x1 or y2 <= y1:
            if verbose:
                print(f"[WARN] Frame {i} has an invalid crop region")
            cropped_faces.append(None)

        crop = frame_rgb[y1:y2, x1:x2].copy()
        face = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        cropped_faces.append(face)

    # Check whether all frames were cropped successfully
    if any(f is None for f in cropped_faces):
        if verbose:
            print("[WARN] Some face crops failed; heart-rate estimation cannot be performed")
        return {"normalized_bvp": None, "bpm": None}

    # Convert to a NumPy array
    cropped_faces_np = np.array(cropped_faces)  # (INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE, 3)

    if verbose:
        print(f"[INFO] Face cropping complete; input shape: {cropped_faces_np.shape}")

    # Initialize the BVPONNX inference engine
    estimator = BVPONNX(
        onnx_model_path=model_path, model_dir=model_dir, verbose=verbose
    )

    # Run BVPONNX inference; only the normalized BVP is returned
    normalized_bvp = estimator.infer(cropped_faces_np)

    if normalized_bvp is not None:
        # Post-process with detrending and band-pass filtering, then estimate heart rate in the frequency domain
        bvp_detrended = _detrend(normalized_bvp, lambda_value=100)
        bvp_filtered = bandpass_filter(bvp_detrended, fs=fps)
        bpm = calculate_hr(bvp_filtered, fs=fps)
    else:
        bpm = None

    estimator.cleanup()

    result = {
        "normalized_bvp": normalized_bvp,
        "bpm": bpm,
    }

    if verbose and bpm is not None:
        print(f"[INFO] Heart-rate estimate: BPM = {bpm:.1f}")

    return result


# ============================================================
#  Standalone usage examples / tests
# ============================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="RhythmFormer ONNX Heart Rate Estimation SDK")
    parser.add_argument("--mode", type=str, choices=["video", "array", "bench"],
                        default="bench", help="Execution mode")
    parser.add_argument("--video", type=str, default=None, help="Video file path (video mode)")
    parser.add_argument("--model", type=str, default=None, help="ONNX Model paths")
    parser.add_argument("--fps", type=int, default=30, help="VideoFrame rate")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    print("=" * 60)
    print("  RhythmFormer ONNX Heart Rate Estimation SDK")
    print("=" * 60)

    if args.mode == "bench":
        # Benchmark mode
        print("\n[INFO] Benchmark mode")
        estimator = BVPONNX(onnx_model_path=args.model, verbose=args.verbose)

        iters = 100
        random_input = np.random.randint(0, 256,
                                         (INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS),
                                         dtype=np.uint8)

        total_time = 0
        for _ in range(iters):
            t1 = time.time()
            bvp = estimator.infer(random_input)
            total_time += time.time() - t1

        print(f"\n[INFO] Benchmark ({iters} inference runs):")
        print(f"  Average time: {total_time/iters*1000:.1f} ms")
        print(f"  BVP output shape: {bvp.shape if bvp is not None else 'None'}")

        estimator.cleanup()

    elif args.mode == "video":
        if args.video is None:
            print("[ERROR] Video mode requires --video")
            exit(1)

        print(f"\n[INFO] Video mode: {args.video}")
        result = estimate_video_hr(
            video_path=args.video, fps=args.fps,
            model_path=args.model, verbose=args.verbose
        )
        if result["avg_hr"] is not None:
            print(f"\n[INFO] Video heart-rate statistics:")
            print(f"  Average heart rate: {result['avg_hr']:.1f} bpm")
            print(f"  Minimum heart rate: {result['min_hr']:.1f} bpm")
            print(f"  Maximum heart rate: {result['max_hr']:.1f} bpm")
            print(f"  Standard deviation:   {result['std_hr']:.1f} bpm")

    elif args.mode == "array":
        print("\n[INFO] Array mode (random-data demonstration)")
        random_frames = np.random.randint(0, 256,
                                          (INPUT_FRAMES, INPUT_SIZE, INPUT_SIZE, 3),
                                          dtype=np.uint8)

        # random_frames = np.load('../video1_chunk0_input.npy')

        print(f'test frame shape: {random_frames.shape}')

        result = estimate_array_hr(
            random_frames, fps=args.fps,
            model_path=args.model, verbose=args.verbose
        )
        bpm_str = f"{result['bpm']:.1f}" if result['bpm'] is not None else "N/A"
        print(f"\n[INFO] Heart rate: {bpm_str} bpm")
