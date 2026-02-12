"""Helper utilities for VIGIL-RAG database operations.

This module provides utility functions for:
- Converting chunking results to database records
- Managing multi-scale clip storage
- Video metadata extraction and storage
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from .chunking_service import ChunkingResult, Chunk

logger = logging.getLogger(__name__)


def compute_video_checksum(video_path: Path) -> str:
    """Compute SHA-256 checksum of a video file.

    Args:
        video_path: Path to video file

    Returns:
        Hex-encoded SHA-256 checksum
    """
    sha256 = hashlib.sha256()
    with open(video_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_video_metadata(video_path: Path) -> dict:
    """Extract metadata from a video file using OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with keys: duration_sec, fps, width, height

    Raises:
        ValueError: If video cannot be opened
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_sec = total_frames / fps if fps > 0 else 0.0

    return {
        "duration_sec": duration_sec,
        "fps": fps,
        "width": width,
        "height": height,
    }


def chunk_to_clip_record(
    chunk: Chunk,
    video_id: str,
    keyframe_dir: Path | None = None,
) -> dict:
    """Convert a Chunk to a database clip record.

    Args:
        chunk: Chunk object
        video_id: UUID of the parent video (as string)
        keyframe_dir: Optional directory where keyframes are stored

    Returns:
        Dictionary suitable for inserting into clips table
    """
    # Build keyframe paths if directory is provided
    keyframe_paths: list[str] | None = None
    if keyframe_dir is not None:
        keyframe_paths = [
            str(keyframe_dir / f"{chunk.level}_frame_{frame_idx:08d}.jpg")
            for frame_idx in chunk.keyframe_indices
        ]

    return {
        "video_id": video_id,
        "level": chunk.level,
        "start_sec": chunk.start_sec,
        "end_sec": chunk.end_sec,
        "keyframe_paths": keyframe_paths,
        "transcript_text": None,  # To be filled by Whisper later
        "embedding_id": None,  # To be filled after embedding extraction
        "clip_idx": None,  # Not used for VIGIL-RAG chunks
    }


def chunking_result_to_clip_records(
    result: ChunkingResult,
    video_id: str,
    keyframe_dir: Path | None = None,
) -> list[dict]:
    """Convert a ChunkingResult to a list of clip records.

    Args:
        result: ChunkingResult from chunking service
        video_id: UUID of the parent video (as string)
        keyframe_dir: Optional directory where keyframes are stored

    Returns:
        List of dictionaries suitable for inserting into clips table
    """
    records: list[dict] = []

    # Add all shots
    for chunk in result.shots:
        records.append(chunk_to_clip_record(chunk, video_id, keyframe_dir))

    # Add all micro chunks
    for chunk in result.micro:
        records.append(chunk_to_clip_record(chunk, video_id, keyframe_dir))

    # Add all meso chunks
    for chunk in result.meso:
        records.append(chunk_to_clip_record(chunk, video_id, keyframe_dir))

    # Add all macro chunks
    for chunk in result.macro:
        records.append(chunk_to_clip_record(chunk, video_id, keyframe_dir))

    logger.info(
        "Generated %d clip records (shots=%d, micro=%d, meso=%d, macro=%d)",
        len(records),
        len(result.shots),
        len(result.micro),
        len(result.meso),
        len(result.macro),
    )

    return records


def create_video_record(
    *,
    file_path: Path,
    domain: str,
    task_id: str | None = None,
    role: str | None = None,
    site_id: str | None = None,
    camera_id: str | None = None,
    operator_id_hash: str | None = None,
    uri: str | None = None,
    storage_path: str | None = None,
    compute_checksum: bool = True,
) -> dict:
    """Create a video record for the videos table.

    Args:
        file_path: Path to video file
        domain: Domain classification (factory, surveillance, sports)
        task_id: Optional task ID (SOPilot compatibility)
        role: Optional role (SOPilot compatibility)
        site_id: Optional site identifier
        camera_id: Optional camera identifier
        operator_id_hash: Optional hashed operator ID
        uri: Optional S3/HTTP URI
        storage_path: Optional object storage path
        compute_checksum: Whether to compute SHA-256 checksum (default True)

    Returns:
        Dictionary suitable for inserting into videos table

    Raises:
        ValueError: If video file doesn't exist or cannot be read
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Video file not found: {file_path}")

    # Extract video metadata
    metadata = extract_video_metadata(file_path)

    # Compute checksum if requested
    checksum = None
    if compute_checksum:
        logger.info("Computing checksum for %s", file_path)
        checksum = compute_video_checksum(file_path)

    return {
        # SOPilot compatibility fields
        "file_path": str(file_path),
        "task_id": task_id,
        "role": role,
        "site_id": site_id,
        "camera_id": camera_id,
        "operator_id_hash": operator_id_hash,
        "num_clips": None,  # To be filled after chunking
        "embedding_model": None,  # To be filled after embedding
        # VIGIL-RAG fields
        "uri": uri,
        "storage_path": storage_path,
        "duration_sec": metadata["duration_sec"],
        "fps": metadata["fps"],
        "width": metadata["width"],
        "height": metadata["height"],
        "domain": domain,
        "checksum": checksum,
    }
