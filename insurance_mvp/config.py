"""Configuration Management for Insurance MVP

Loads configuration from multiple sources with priority:
1. Environment variables (highest priority)
2. YAML config file
3. Default values (lowest priority)
"""

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class DeviceType(str, Enum):
    """Device type for model inference"""

    CUDA = "cuda"
    CPU = "cpu"
    AUTO = "auto"


class CosmosBackend(str, Enum):
    """Video-LLM backend"""

    QWEN25VL = "qwen2.5-vl-7b"
    MOCK = "mock"


class WhisperBackend(str, Enum):
    """Whisper transcription backend"""

    OPENAI_WHISPER = "openai-whisper"
    MOCK = "mock"


@dataclass
class VideoConfig:
    """Video processing configuration"""

    max_resolution: tuple = (1280, 720)  # Max resolution before downsampling
    fps_sampling: int = 2  # Extract N frames per second
    chunk_duration_sec: float = 5.0  # Chunk duration for mining
    min_clip_duration_sec: float = 2.0  # Minimum clip duration
    max_clip_duration_sec: float = 30.0  # Maximum clip duration


@dataclass
class MiningConfig:
    """B1: Signal mining configuration"""

    top_k_clips: int = 20  # Top K dangerous clips to extract

    # Audio thresholds
    audio_brake_threshold: float = 0.7
    audio_horn_threshold: float = 0.6
    audio_crash_threshold: float = 0.8

    # Motion thresholds
    motion_magnitude_threshold: float = 50.0  # Optical flow magnitude
    motion_suddenness_threshold: float = 0.7  # Sudden changes

    # Proximity thresholds
    proximity_near_distance_m: float = 5.0  # Close object distance
    proximity_confidence_threshold: float = 0.5  # YOLO confidence

    # Fusion weights
    audio_weight: float = 0.3
    motion_weight: float = 0.3
    proximity_weight: float = 0.4

    # Clip extraction (ffmpeg)
    extract_clips: bool = False  # Extract danger clips as separate video files


@dataclass
class CosmosConfig:
    """B2: Video-LLM inference configuration"""

    backend: CosmosBackend = CosmosBackend.QWEN25VL
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: DeviceType = DeviceType.AUTO

    # Inference settings
    max_new_tokens: int = 512
    temperature: float = 0.1  # Low temperature for factual assessment
    top_p: float = 0.9

    # Video processing
    max_pixels: int = 602112  # Qwen2.5-VL hard limit (768*28*28)
    fps: int = 2  # Frames per second for VLM

    # Batch processing
    max_concurrent_inferences: int = 2  # GPU memory limit


@dataclass
class ConformalConfig:
    """B4: Conformal prediction configuration"""

    alpha: float = 0.1  # 90% confidence (1 - alpha)
    severity_levels: list[str] = field(default_factory=lambda: ["NONE", "LOW", "MEDIUM", "HIGH"])

    # Calibration
    use_pretrained_calibration: bool = True
    calibration_data_path: str | None = None


@dataclass
class WhisperConfig:
    """Whisper transcription configuration"""

    backend: WhisperBackend = WhisperBackend.OPENAI_WHISPER
    model_size: str = "base"  # tiny, base, small, medium, large
    language: str = "ja"  # Japanese for dashcam videos
    device: DeviceType = DeviceType.AUTO


@dataclass
class FaultConfig:
    """Fault assessment configuration (mirrors FaultAssessmentConfig)"""

    # Scenario-specific fault ratios
    rear_end_default: float = 100.0
    rear_end_sudden_stop: float = 70.0
    head_on_default: float = 50.0
    side_swipe_lane_change: float = 80.0
    side_swipe_unknown: float = 50.0
    left_turn_default: float = 75.0
    intersection_no_signal: float = 50.0
    # Adjustments
    red_light_violation_fault: float = 100.0
    excessive_speed_adjustment: float = 10.0
    weather_adjustment: float = 5.0
    # Thresholds
    excessive_speed_threshold_kmh: float = 20.0
    min_fault_ratio: float = 0.0
    max_fault_ratio: float = 100.0
    # Speed limits (assumed when no posted limit available)
    speed_limit_urban_kmh: float = 60.0
    speed_trigger_kmh: float = 80.0
    max_speed_penalty_pct: float = 15.0


@dataclass
class FraudConfig:
    """Fraud detection configuration (mirrors FraudDetectionConfig)"""

    # Thresholds
    high_risk_threshold: float = 0.65
    medium_risk_threshold: float = 0.4
    # Indicator weights
    weight_audio_visual_mismatch: float = 0.25
    weight_damage_inconsistency: float = 0.20
    weight_suspicious_positioning: float = 0.15
    weight_claim_history: float = 0.20
    weight_claim_amount_anomaly: float = 0.10
    weight_timing_anomaly: float = 0.10
    # Claim history thresholds
    suspicious_claims_per_year: int = 3
    suspicious_claims_per_month: int = 2
    claim_cluster_days: int = 30
    # Amount thresholds
    claim_amount_outlier_threshold: float = 3.0
    # Speed/damage consistency
    min_speed_for_damage_kmh: float = 15.0
    max_speed_no_damage_kmh: float = 10.0
    # Reporting timing
    suspicious_delay_hours: float = 72.0
    suspicious_quick_report_hours: float = 0.5


@dataclass
class ApiConfig:
    """FastAPI application configuration"""

    database_url: str = "sqlite:///./insurance.db"
    database_echo: bool = False
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = 500
    allowed_extensions: list[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    worker_max_threads: int = 4
    use_pipeline: bool = False
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    api_version: str = "1.0.0"
    api_title: str = "Insurance MVP API"
    dev_mode: bool = True


@dataclass
class VlmConfig:
    """Video-LLM inference configuration (mirrors VLMConfig in cosmos/client.py)"""

    fps: float = 2.0
    max_frames: int = 48
    max_new_tokens: int = 512
    temperature: float = 0.1
    timeout_sec: float = 300.0
    jpeg_quality: int = 75
    max_clip_duration_sec: float = 60.0
    enable_cpu_fallback: bool = True
    gpu_cleanup: bool = True
    frame_extraction_timeout_sec: float = 120.0


@dataclass
class PipelineConfig:
    """Pipeline orchestration configuration"""

    # Input/Output
    video_path: str | None = None
    video_dir: str | None = None
    output_dir: str = "results"

    # Processing
    parallel_workers: int = 1  # Number of videos to process in parallel
    resume_from_checkpoint: bool = True  # Skip already processed clips
    save_intermediate_results: bool = True  # Save after each stage

    # Error handling
    continue_on_error: bool = True  # Continue processing on failure
    max_retries: int = 3
    retry_delay_sec: float = 5.0

    # Logging
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_file: str | None = None  # If None, log to console only

    # Component configs
    video: VideoConfig = field(default_factory=VideoConfig)
    mining: MiningConfig = field(default_factory=MiningConfig)
    cosmos: CosmosConfig = field(default_factory=CosmosConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    fraud: FraudConfig = field(default_factory=FraudConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)

    # Feature flags
    enable_transcription: bool = True
    enable_conformal: bool = True
    enable_fraud_detection: bool = True
    enable_fault_assessment: bool = True
    enable_recalibration: bool = True

    # Performance monitoring
    enable_profiling: bool = False
    enable_metrics: bool = True


class ConfigLoader:
    """Loads configuration from YAML and environment variables"""

    @staticmethod
    def load_from_yaml(yaml_path: str) -> dict[str, Any]:
        """Load configuration from YAML file"""
        yaml_path_obj = Path(yaml_path)
        if not yaml_path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path_obj, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        return config_dict or {}

    @staticmethod
    def load_from_env() -> dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}

        # Map environment variables to config structure
        env_mappings = {
            "INSURANCE_OUTPUT_DIR": ("output_dir", str),
            "INSURANCE_LOG_LEVEL": ("log_level", str),
            "INSURANCE_LOG_FILE": ("log_file", str),
            "INSURANCE_PARALLEL_WORKERS": ("parallel_workers", int),
            "INSURANCE_CONTINUE_ON_ERROR": ("continue_on_error", lambda x: x.lower() == "true"),
            # Cosmos
            "INSURANCE_COSMOS_BACKEND": ("cosmos.backend", str),
            "INSURANCE_COSMOS_DEVICE": ("cosmos.device", str),
            "INSURANCE_COSMOS_MAX_CONCURRENT": ("cosmos.max_concurrent_inferences", int),
            # Mining
            "INSURANCE_MINING_TOP_K": ("mining.top_k_clips", int),
            # Conformal
            "INSURANCE_CONFORMAL_ALPHA": ("conformal.alpha", float),
            # Whisper
            "INSURANCE_WHISPER_BACKEND": ("whisper.backend", str),
            "INSURANCE_WHISPER_MODEL": ("whisper.model_size", str),
            "INSURANCE_WHISPER_DEVICE": ("whisper.device", str),
            # Feature flags
            "INSURANCE_ENABLE_TRANSCRIPTION": ("enable_transcription", lambda x: x.lower() == "true"),
            "INSURANCE_ENABLE_CONFORMAL": ("enable_conformal", lambda x: x.lower() == "true"),
            "INSURANCE_ENABLE_FRAUD": ("enable_fraud_detection", lambda x: x.lower() == "true"),
            "INSURANCE_ENABLE_PROFILING": ("enable_profiling", lambda x: x.lower() == "true"),
        }

        for env_var, (config_key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    # Support nested keys like "cosmos.backend"
                    keys = config_key.split(".")
                    current = env_config
                    for key in keys[:-1]:
                        if key not in current:
                            current[key] = {}
                        current = current[key]
                    current[keys[-1]] = converted_value
                except (ValueError, TypeError) as e:
                    print(f"Warning: Failed to convert env var {env_var}={value}: {e}")

        return env_config

    @staticmethod
    def merge_configs(*configs: dict[str, Any]) -> dict[str, Any]:
        """Merge multiple config dictionaries (later configs override earlier ones)"""
        merged = {}

        for config in configs:
            merged = ConfigLoader._deep_merge(merged, config)

        return merged

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    @staticmethod
    def dict_to_config(config_dict: dict[str, Any]) -> PipelineConfig:
        """Convert dictionary to PipelineConfig dataclass"""

        # Extract nested configs
        video_dict = config_dict.pop("video", {})
        mining_dict = config_dict.pop("mining", {})
        cosmos_dict = config_dict.pop("cosmos", {})
        conformal_dict = config_dict.pop("conformal", {})
        whisper_dict = config_dict.pop("whisper", {})
        fault_dict = config_dict.pop("fault", {})
        fraud_dict = config_dict.pop("fraud", {})
        api_dict = config_dict.pop("api", {})
        vlm_dict = config_dict.pop("vlm", {})

        # Convert enum strings to enum values
        if "backend" in cosmos_dict:
            cosmos_dict["backend"] = CosmosBackend(cosmos_dict["backend"])
        if "device" in cosmos_dict:
            cosmos_dict["device"] = DeviceType(cosmos_dict["device"])
        if "backend" in whisper_dict:
            whisper_dict["backend"] = WhisperBackend(whisper_dict["backend"])
        if "device" in whisper_dict:
            whisper_dict["device"] = DeviceType(whisper_dict["device"])

        # Create config objects
        video_config = VideoConfig(**video_dict)
        mining_config = MiningConfig(**mining_dict)
        cosmos_config = CosmosConfig(**cosmos_dict)
        conformal_config = ConformalConfig(**conformal_dict)
        whisper_config = WhisperConfig(**whisper_dict)
        fault_config = FaultConfig(**fault_dict)
        fraud_config = FraudConfig(**fraud_dict)
        api_config = ApiConfig(**api_dict)
        vlm_config = VlmConfig(**vlm_dict)

        # Create main config
        config = PipelineConfig(
            **config_dict,
            video=video_config,
            mining=mining_config,
            cosmos=cosmos_config,
            conformal=conformal_config,
            whisper=whisper_config,
            fault=fault_config,
            fraud=fraud_config,
            api=api_config,
            vlm=vlm_config,
        )

        return config


def load_config(yaml_path: str | None = None, override_dict: dict[str, Any] | None = None) -> PipelineConfig:
    """
    Load configuration from multiple sources.

    Priority (highest to lowest):
    1. override_dict (programmatic overrides)
    2. Environment variables
    3. YAML file
    4. Default values

    Args:
        yaml_path: Path to YAML config file (optional)
        override_dict: Programmatic overrides (optional)

    Returns:
        PipelineConfig instance
    """
    loader = ConfigLoader()

    # Start with defaults
    default_config = asdict(PipelineConfig())

    # Load from YAML if provided
    yaml_config = {}
    if yaml_path:
        try:
            yaml_config = loader.load_from_yaml(yaml_path)
        except FileNotFoundError as e:
            print(f"Warning: {e}")

    # Load from environment variables
    env_config = loader.load_from_env()

    # Merge configs (later overrides earlier)
    merged_config = loader.merge_configs(default_config, yaml_config, env_config, override_dict or {})

    # Convert to PipelineConfig
    config = loader.dict_to_config(merged_config)

    return config


def save_config(config: PipelineConfig, yaml_path: str):
    """Save configuration to YAML file"""
    config_dict = asdict(config)

    # Convert enums and tuples to YAML-friendly format
    def convert_for_yaml(obj):
        if isinstance(obj, dict):
            return {k: convert_for_yaml(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_yaml(v) for v in obj]
        elif isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    config_dict = convert_for_yaml(config_dict)

    yaml_path_obj = Path(yaml_path)
    yaml_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(yaml_path_obj, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
