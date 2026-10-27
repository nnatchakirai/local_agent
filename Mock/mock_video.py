# ComfyUI Cloud 영상 생성 모의 API (Phase 1: 추천 목록과 모의 작업)
import time
import uuid

from devlog import dev_log

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# 실제 파이프라인 단계(제안 문서의 상태 모델)를 시간 경과로 흉내낸다.
STAGES = [
    ("validating", 2),
    ("background_generation", 5),
    ("video_generation", 8),
    ("publishing", 10),
]
MOCK_VIDEO_URL = "/static/sample.mp4"

tasks: dict[str, dict] = {}


class VideoRequest(BaseModel):
    place_id: str
    session_id: str = ""
    character_asset_id: str = "character_default_v1"


STAGE_DESC = {
    "validating": "입력/장소/LoRA 매핑 확인 중",
    "background_generation": "장소 LoRA로 배경 이미지 생성 중",
    "video_generation": "character.png와 배경으로 이미지→영상 변환 중",
    "publishing": "결과 영상 검증 및 저장 중",
    "completed": "완료 - 샘플 영상(sample.mp4)을 결과로 전달",
}


def _log_stage(task_id: str, task: dict, stage: str) -> None:
    """단계가 바뀔 때만 한 번 기록한다 (폴링 때마다 중복 출력 방지)."""
    if task["last_stage"] != stage:
        task["last_stage"] = stage
        dev_log(f"[모의 ComfyUI] {task_id}: {STAGE_DESC[stage]}")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_or_create_task(place_id: str, session_id: str) -> tuple[str, bool]:
    """동일 설문 세션·장소의 진행 중/완료 작업은 재사용한다. (task_id, reused) 반환"""
    for task_id, t in tasks.items():
        if t["place_id"] == place_id and t["session_id"] == session_id:
            return task_id, True

    task_id = f"task_{uuid.uuid4().hex[:8]}"
    tasks[task_id] = {"place_id": place_id, "session_id": session_id,
                      "started": time.time(), "created_at": _now(), "last_stage": None}
    dev_log(f"[모의 ComfyUI] 영상 생성 작업 등록: {task_id} (장소 {place_id}) - 실제 ComfyUI 호출 없이 시간 경과로 단계만 흉내냅니다")
    return task_id, False


@router.post("/api/videos")
async def create_video(req: VideoRequest):
    task_id, reused = get_or_create_task(req.place_id, req.session_id)
    return {"task_id": task_id, "place_id": req.place_id, "reused": reused,
            "status_url": f"/api/video-status/{task_id}"}


@router.get("/api/video-status/{task_id}")
async def video_status(task_id: str):
    task = tasks.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")

    elapsed = time.time() - task["started"]
    base = {"task_id": task_id, "place_id": task["place_id"],
            "created_at": task["created_at"], "updated_at": _now(),
            "error": None}

    for stage, until in STAGES:
        if elapsed < until:
            _log_stage(task_id, task, stage)
            return {**base, "status": "queued" if elapsed < 0.5 else "running",
                    "stage": stage, "result": None}

    _log_stage(task_id, task, "completed")
    return {**base, "status": "succeeded", "stage": "completed",
            "result": {"video_url": MOCK_VIDEO_URL, "duration_seconds": 10.0}}
