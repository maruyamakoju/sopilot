"""Video upload, listing, streaming, and management endpoints."""

from fastapi import APIRouter, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from sopilot.api.error_handling import service_errors
from sopilot.api.routes._helpers import get_service
from sopilot.core.video_quality import VideoQualityChecker
from sopilot.schemas import (
    VideoDetailResponse,
    VideoIngestResponse,
    VideoListItem,
    VideoListResponse,
    VideoUpdateRequest,
)


def build_video_router() -> APIRouter:
    router = APIRouter(tags=["Videos"])

    @router.post("/videos", response_model=VideoIngestResponse, summary="Upload a trainee video")
    @service_errors
    async def upload_video(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        task_id: str = Form(...),
        site_id: str | None = Form(default=None),
        camera_id: str | None = Form(default=None),
        operator_id_hash: str | None = Form(default=None),
        recorded_at: str | None = Form(default=None),
    ) -> VideoIngestResponse:
        service = get_service(request)
        payload = await run_in_threadpool(
            service.ingest_video,
            original_filename=file.filename or "upload",
            file_obj=file.file,
            task_id=task_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
            is_gold=False,
        )
        return VideoIngestResponse(**payload)

    @router.post("/gold", response_model=VideoIngestResponse, summary="Upload a gold (reference) video")
    @service_errors
    async def upload_gold(
        request: Request,
        file: UploadFile = File(...),  # noqa: B008
        task_id: str = Form(...),
        site_id: str | None = Form(default=None),
        camera_id: str | None = Form(default=None),
        operator_id_hash: str | None = Form(default=None),
        recorded_at: str | None = Form(default=None),
        enforce_quality: bool = Form(default=False),
    ) -> VideoIngestResponse:
        service = get_service(request)
        payload = await run_in_threadpool(
            service.ingest_video,
            original_filename=file.filename or "upload",
            file_obj=file.file,
            task_id=task_id,
            site_id=site_id,
            camera_id=camera_id,
            operator_id_hash=operator_id_hash,
            recorded_at=recorded_at,
            is_gold=True,
            enforce_quality=enforce_quality,
        )
        return VideoIngestResponse(**payload)

    @router.get("/videos", response_model=VideoListResponse, summary="List videos")
    @service_errors
    async def list_videos(
        request: Request,
        site_id: str | None = Query(default=None),
        is_gold: bool | None = Query(default=None),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> VideoListResponse:
        service = get_service(request)
        payload = await run_in_threadpool(
            service.list_videos, site_id=site_id, is_gold=is_gold, limit=limit,
        )
        total = await run_in_threadpool(
            service.count_videos, site_id=site_id, is_gold=is_gold,
        )
        items = [VideoListItem(**item) for item in payload]
        return VideoListResponse(items=items, total=total, has_more=len(items) >= limit)

    @router.get("/videos/{video_id}", response_model=VideoDetailResponse, summary="Get video details")
    @service_errors
    async def get_video(request: Request, video_id: int) -> VideoDetailResponse:
        service = get_service(request)
        payload = await run_in_threadpool(service.get_video_detail, video_id=video_id)
        return VideoDetailResponse(**payload)

    @router.patch("/videos/{video_id}", response_model=VideoDetailResponse, summary="Update video metadata")
    @service_errors
    async def update_video(request: Request, video_id: int, body: VideoUpdateRequest) -> VideoDetailResponse:
        service = get_service(request)
        payload = await run_in_threadpool(
            service.update_video_metadata,
            video_id,
            site_id=body.site_id,
            camera_id=body.camera_id,
            operator_id_hash=body.operator_id_hash,
            recorded_at=body.recorded_at,
        )
        return VideoDetailResponse(**payload)

    @router.delete("/videos/{video_id}", summary="Delete a video")
    @service_errors
    async def delete_video(
        request: Request,
        video_id: int,
        force: bool = Query(default=False, description="Also delete associated score jobs and reviews"),
    ) -> dict:
        service = get_service(request)
        return await run_in_threadpool(service.delete_video, video_id=video_id, force=force)

    @router.get("/videos/{video_id}/stream", summary="Stream video file")
    @service_errors
    async def stream_video(request: Request, video_id: int) -> FileResponse:
        service = get_service(request)
        path = await run_in_threadpool(service.get_video_stream_path, video_id=video_id)
        return FileResponse(path=path)

    @router.get("/videos/{video_id}/quality", tags=["Videos"], summary="Run quality check on video")
    @service_errors
    async def check_video_quality(request: Request, video_id: int) -> dict:
        service = get_service(request)
        path = await run_in_threadpool(service.get_video_stream_path, video_id=video_id)
        checker = VideoQualityChecker()
        report = await run_in_threadpool(checker.check, path)
        return report.to_dict()

    return router
