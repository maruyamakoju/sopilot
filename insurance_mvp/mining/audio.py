"""Audio Signal Analysis for Insurance Video Review

Analyzes audio track to detect danger signals:
- Sudden volume changes (brake sounds, crashes)
- Horn detection via frequency analysis
- Impact sounds

Design:
- Process audio in 1-second windows
- Return normalized danger scores [0.0, 1.0]
- Handle missing audio gracefully
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio analysis"""

    # RMS (root mean square) analysis
    rms_window_sec: float = 1.0  # Window size for RMS computation
    rms_percentile: float = 90.0  # Percentile for normalization
    rms_delta_threshold: float = 2.0  # Multiplier for sudden change detection

    # Horn detection (300-1000 Hz band)
    horn_freq_min: int = 300  # Hz
    horn_freq_max: int = 1000  # Hz
    horn_power_threshold: float = 0.7  # Relative power threshold

    # Sampling
    sample_rate: int = 16000  # Target sample rate for processing


class AudioAnalyzer:
    """
    Analyze audio track for danger signals.

    Extracts:
    1. RMS (volume level) - baseline loudness
    2. Delta-RMS - sudden volume changes (crashes, hard braking)
    3. Horn-band FFT - horn detection via frequency analysis
    """

    def __init__(self, config: AudioConfig | None = None):
        self.config = config or AudioConfig()

    def analyze(self, video_path: Path | str) -> np.ndarray:
        """
        Analyze audio track and return danger scores per second.

        Args:
            video_path: Path to video file

        Returns:
            np.ndarray: Danger scores per second [0.0, 1.0], shape (n_seconds,)

        Raises:
            RuntimeError: If video file cannot be read
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise RuntimeError(f"Video file not found: {video_path}")

        logger.info(f"Analyzing audio: {video_path.name}")

        # Extract audio samples
        try:
            audio_samples, sample_rate, duration_sec = self._extract_audio(video_path)
        except Exception as e:
            logger.warning(f"Failed to extract audio from {video_path.name}: {e}")
            # Return zero scores (no audio = no audio danger)
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps > 0:
                duration_sec = int(np.ceil(frame_count / fps))
            else:
                duration_sec = 0

            logger.info(f"No audio track, returning zeros for {duration_sec} seconds")
            return np.zeros(duration_sec, dtype=np.float32)

        n_seconds = int(np.ceil(duration_sec))

        # Compute RMS per second
        rms_scores = self._compute_rms(audio_samples, sample_rate, n_seconds)

        # Compute delta-RMS (sudden changes)
        delta_rms_scores = self._compute_delta_rms(rms_scores)

        # Compute horn-band power (frequency analysis)
        horn_scores = self._compute_horn_band(audio_samples, sample_rate, n_seconds)

        # Fuse signals (max of all signals)
        danger_scores = np.maximum.reduce([delta_rms_scores, horn_scores])

        logger.info(
            f"Audio analysis complete: {n_seconds}s, "
            f"mean_danger={danger_scores.mean():.3f}, "
            f"max_danger={danger_scores.max():.3f}, "
            f"peaks={np.sum(danger_scores > 0.5)}"
        )

        return danger_scores

    def _extract_audio(self, video_path: Path) -> tuple[np.ndarray, int, float]:
        """
        Extract audio samples from video file.

        Uses OpenCV to read video, then extracts audio via ffmpeg if available.
        Fallback: return silent audio if extraction fails.

        Returns:
            Tuple of (samples, sample_rate, duration_sec)
        """
        # Try to extract audio using cv2.VideoCapture
        # Note: cv2 doesn't directly support audio extraction
        # We use a subprocess call to ffmpeg for robust audio extraction

        import subprocess
        import tempfile

        try:
            # Create temp file for audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            # Extract audio with ffmpeg
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",  # No video
                "-acodec",
                "pcm_s16le",  # PCM 16-bit
                "-ar",
                str(self.config.sample_rate),  # Sample rate
                "-ac",
                "1",  # Mono
                "-y",  # Overwrite
                tmp_path,
                "-loglevel",
                "error",
            ]

            result = subprocess.run(cmd, capture_output=True, timeout=60)

            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

            # Read WAV file
            import wave

            with wave.open(tmp_path, "rb") as wav:
                sample_rate = wav.getframerate()
                n_frames = wav.getnframes()
                raw_bytes = wav.readframes(n_frames)

                # Convert to float32 [-1.0, 1.0]
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                samples /= 32768.0

                duration_sec = n_frames / sample_rate

            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)

            return samples, sample_rate, duration_sec

        except Exception as e:
            logger.warning(f"Audio extraction failed: {e}, returning silent audio")
            # Get video duration from metadata
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps > 0:
                duration_sec = frame_count / fps
            else:
                duration_sec = 10.0  # Fallback

            # Return silent audio
            n_samples = int(duration_sec * self.config.sample_rate)
            samples = np.zeros(n_samples, dtype=np.float32)

            return samples, self.config.sample_rate, duration_sec

    def _compute_rms(self, samples: np.ndarray, sample_rate: int, n_seconds: int) -> np.ndarray:
        """
        Compute RMS (root mean square) energy per second.

        RMS measures overall loudness - useful for detecting loud events.
        """
        window_size = int(sample_rate * self.config.rms_window_sec)
        rms_values = []

        for sec in range(n_seconds):
            start_idx = sec * sample_rate
            end_idx = min(start_idx + window_size, len(samples))

            if end_idx <= start_idx:
                rms_values.append(0.0)
                continue

            window_samples = samples[start_idx:end_idx]
            rms = np.sqrt(np.mean(window_samples**2))
            rms_values.append(rms)

        rms_array = np.array(rms_values, dtype=np.float32)

        # Normalize using percentile (avoid outlier saturation)
        if len(rms_array) > 0 and rms_array.max() > 0:
            percentile_val = np.percentile(rms_array, self.config.rms_percentile)
            if percentile_val > 0:
                rms_array = rms_array / percentile_val
                rms_array = np.clip(rms_array, 0.0, 1.0)

        return rms_array

    def _compute_delta_rms(self, rms_scores: np.ndarray) -> np.ndarray:
        """
        Compute sudden RMS changes (delta-RMS).

        Detects abrupt volume spikes (brake sounds, crashes, impacts).
        """
        if len(rms_scores) < 2:
            return rms_scores.copy()

        # Compute first-order difference
        delta = np.diff(rms_scores, prepend=rms_scores[0])

        # Only keep positive changes (volume increases)
        delta = np.maximum(delta, 0.0)

        # Normalize
        if delta.max() > 0:
            # Use threshold-based normalization
            threshold = delta.mean() + self.config.rms_delta_threshold * delta.std()
            delta = delta / max(threshold, 1e-6)
            delta = np.clip(delta, 0.0, 1.0)

        return delta

    def _compute_horn_band(self, samples: np.ndarray, sample_rate: int, n_seconds: int) -> np.ndarray:
        """
        Detect horn sounds via FFT frequency analysis.

        Car horns typically emit 300-1000 Hz tones. Compute power in this band.
        """
        window_size = int(sample_rate * self.config.rms_window_sec)
        horn_scores = []

        for sec in range(n_seconds):
            start_idx = sec * sample_rate
            end_idx = min(start_idx + window_size, len(samples))

            if end_idx <= start_idx:
                horn_scores.append(0.0)
                continue

            window_samples = samples[start_idx:end_idx]

            # Apply Hanning window to reduce spectral leakage
            window_samples = window_samples * np.hanning(len(window_samples))

            # Compute FFT
            fft = np.fft.rfft(window_samples)
            fft_freqs = np.fft.rfftfreq(len(window_samples), 1.0 / sample_rate)
            fft_power = np.abs(fft) ** 2

            # Extract horn band (300-1000 Hz)
            horn_mask = (fft_freqs >= self.config.horn_freq_min) & (fft_freqs <= self.config.horn_freq_max)
            horn_power = fft_power[horn_mask].sum()
            total_power = fft_power.sum()

            # Relative power in horn band
            if total_power > 0:
                horn_ratio = horn_power / total_power
            else:
                horn_ratio = 0.0

            # Threshold-based detection
            if horn_ratio >= self.config.horn_power_threshold:
                horn_scores.append(horn_ratio)
            else:
                horn_scores.append(0.0)

        horn_array = np.array(horn_scores, dtype=np.float32)

        # Normalize
        if horn_array.max() > 0:
            horn_array = horn_array / horn_array.max()

        return horn_array
