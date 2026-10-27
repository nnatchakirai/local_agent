# 개발용 진행 로그: 터미널 출력 + 프론트엔드 개발 패널 조회용 메모리 버퍼
import time

logs: list[dict] = []


def dev_log(message: str) -> None:
    entry = {"id": len(logs), "time": time.strftime("%H:%M:%S"), "message": message}
    logs.append(entry)
    print(f"[진행 {entry['time']}] {message}", flush=True)
