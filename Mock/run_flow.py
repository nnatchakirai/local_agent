#!/usr/bin/env python3
"""
실행 흐름 확인 스크립트 (개발 전용)

실행 중인 서버(127.0.0.1:5000)에 실제 HTTP 요청을 보내면서
- 어떤 API를 호출했는지 (요청/응답 코드/소요 시간)
- 서버가 내부적으로 어떤 단계를 거쳤는지 (/api/dev-log 로 가져온 서버 로그)
- 영상 생성 진행도
를 순서대로 출력한다. app.py 를 import 하지 않고 서버와 HTTP 로만 통신한다.

사용법 (서버를 먼저 켜둔다: python app.py)
    python run_flow.py            # 설문 답변 기본값 0,0,0,0
    python run_flow.py 1,0,1,1    # 문항별 선택지(0/1)를 직접 지정
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:5000"
DEFAULT_ANSWERS = [0, 0, 0, 0]
TOTAL_STEPS = 4

# mock_video.py 의 단계 순서와 동일
VIDEO_STAGES = ["validating", "background_generation", "video_generation", "publishing", "completed"]
POLL_SEC = 1
VIDEO_TIMEOUT_SEC = 60

last_log_id = -1


def request(method: str, path: str, body=None):
    """(status_code, json_body)를 반환한다. 연결 실패는 호출한 쪽에서 처리한다."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        BASE_URL + path, data=data, method=method,
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as res:
            return res.getcode(), json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"raw": raw}


def print_server_logs():
    """서버 dev_log 중 아직 보지 않은 것만 출력한다."""
    global last_log_id
    _, entries = request("GET", f"/api/dev-log?after={last_log_id}")
    for e in entries:
        sys.stdout.write("\r\033[K")
        print(f"    [서버 {e['time']}] {e['message']}")
        last_log_id = e["id"]


def step_header(n: int, title: str):
    print(f"\n[{n}/{TOTAL_STEPS}] {title}")
    print("-" * 60)


def call_api(method: str, path: str, body=None, slow=False):
    """API 한 번을 호출하고 요청/응답을 출력한다. slow=True면 대기 중 서버 로그와 경과 시간을 표시한다."""
    print(f"  → {method} {path}" + (f"  body={json.dumps(body, ensure_ascii=False)}" if body else ""))
    start = time.time()
    result = {}

    def worker():
        try:
            result["value"] = request(method, path, body)
        except Exception as e:  # 연결 실패 등
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while t.is_alive():
        if slow:
            print_server_logs()
            sys.stdout.write(f"\r  ... 서버 처리 중 {int(time.time() - start)}초")
            sys.stdout.flush()
        t.join(1)
    sys.stdout.write("\r\033[K")

    if "error" in result:
        raise result["error"]
    status, data = result["value"]
    print(f"  ← {status} ({time.time() - start:.2f}초)")
    print_server_logs()
    return status, data


def progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = width * done // total
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total}"


def parse_answers() -> list:
    if len(sys.argv) < 2:
        return DEFAULT_ANSWERS
    return [int(x) for x in sys.argv[1].split(",")]


def main():
    answers = parse_answers()
    print("=" * 60)
    print(f" 실행 흐름 확인 | 서버: {BASE_URL} | 설문 답변: {answers}")
    print("=" * 60)

    step_header(1, "설문 문항 조회")
    status, questions = call_api("GET", "/api/survey/questions")
    for i, q in enumerate(questions):
        picked = q["options"][answers[i]] if i < len(answers) and answers[i] in (0, 1) else "?"
        print(f"    Q{i + 1}. {q['question']} → {picked}")

    step_header(2, "설문 제출 → 추천 생성 (LLM 호출, 시간이 걸릴 수 있음)")
    status, survey = call_api("POST", "/api/survey", {"answers": answers}, slow=True)
    if status != 200:
        print(f"  ✗ 실패: {survey}")
        sys.exit(1)
    print(f"  취향 문장: {survey['preference']}")
    for i, item in enumerate(survey["recommendations"], 1):
        print(f"    #{i} {item['place_id']} | {item['place_name']}")

    step_header(3, f"영상 생성 진행도 확인 (task_id={survey['task_id']})")
    deadline = time.time() + VIDEO_TIMEOUT_SEC
    while time.time() < deadline:
        status, v = request("GET", f"/api/video-status/{survey['task_id']}")
        stage = v.get("stage", "")
        done = VIDEO_STAGES.index(stage) + 1 if stage in VIDEO_STAGES else 0
        sys.stdout.write(f"\r\033[K  {progress_bar(done, len(VIDEO_STAGES))} {v.get('status')} / {stage}")
        sys.stdout.flush()
        print_server_logs()
        if v.get("status") in ("succeeded", "failed"):
            break
        time.sleep(POLL_SEC)
    print()
    if v.get("status") == "succeeded":
        print(f"  ✓ 영상 완료: {v['result']['video_url']}")
    else:
        print(f"  ✗ 영상 미완료: status={v.get('status')} error={v.get('error')}")

    step_header(4, "전시 연계 API 조회 (GET /current_region)")
    status, region = call_api("GET", "/current_region?region_id=GIMHAE")
    print(f"  응답 필드: {list(region.keys())}")
    for i, item in enumerate(region.get("recommendations", []), 1):
        print(f"    #{i} {item['place_id']} | {item['place_name']}")

    print("\n" + "=" * 60)
    print(" 완료")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as e:
        print(f"\n[FAIL] 서버 연결 실패: {e}\n  -> 서버가 실행 중인지 확인하세요: python app.py")
        sys.exit(1)
