#!/usr/bin/env python3
"""
[별첨] Re:Local 연계 REST API 명세서 전체 요구 조건 자동 검증 테스트 스크립트

검증 항목:
1. (2) API 기본 명세 검증:
   - Host / Port 5000 바인드 및 연결
   - HTTP GET 프로토콜
   - HTTP 200 OK 응답 상태
   - UTF-8 인코딩 (Content-Type 헤더)
   - 웹 연계를 위한 CORS 헤더 존재 여부
2. (6) 오류 Response 권장 검증:
   - region_id 파라미터 누락 시
   - region_id 파라미터가 빈 값일 때
   - 지원하지 않는 타 지역(예: DAMYANG) 요청 시
   - {"region_id": "...", "message": "..."} 포맷 준수 여부
3. (4), (5) 정상 응답 형식 및 필수 Response Field 검증:
   - Root Field: region_id, recommendations
   - Recommendation POI: place_name, place_id, recommend_reason 필수 포함
   - 최소 5개 이상의 추천 POI 포함 여부
   - 중복 추천(place_id 중복) 여부 검증
   - CSV 기준 데이터와의 장소코드(place_id) 및 장소명(place_name) 100% 일치 검증
   - recommend_reason 문장 수 (2~5문장) 및 종결부호 규격 검증
4. 전시/연계 현장 안정성 검증:
   - 대소문자 무시 지원 ('gimhae' -> 정상 추천)
   - POST /preference 취향 변경 후 즉각 응답 속도 (캐싱 동작 확인)
"""

import sys
import re
import json
import time
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd

BASE_URL = "http://127.0.0.1:5000"
CSV_PATH = "여행지_20_김해_수정본.csv"


def load_csv_places() -> dict:
    """CSV 파일로부터 {place_id: place_name.strip()} 맵을 로드한다."""
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        print(f"[FAIL] CSV 로드 실패: {e}")
        sys.exit(1)

    places = {}
    for _, row in df.iterrows():
        pid = str(row.get("place_id", "")).strip()
        pname = str(row.get("place_name", "")).strip()
        if pid and pname:
            places[pid] = pname
    print(f"[INFO] CSV에서 총 {len(places)}개의 유효 관광지를 로드했습니다.")
    return places


def http_get(url: str, params: dict = None, custom_headers: dict = None):
    """GET 요청을 보내고 (status_code, headers, body_json, elapsed_time)을 반환한다."""
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{url}?{query_string}"

    # 웹 브라우저/키오스크 프론트엔드의 CORS 요청 시뮬레이션을 위해 Origin 헤더 포함
    headers = {"Origin": "http://127.0.0.1:3000"}
    if custom_headers:
        headers.update(custom_headers)

    start_t = time.perf_counter()
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            elapsed = time.perf_counter() - start_t
            status_code = res.getcode()
            headers = dict(res.info())
            raw_data = res.read().decode("utf-8")
            body = json.loads(raw_data)
            return status_code, headers, body, elapsed
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start_t
        status_code = e.code
        headers = dict(e.headers)
        raw_data = e.read().decode("utf-8")
        try:
            body = json.loads(raw_data)
        except Exception:
            body = {"raw": raw_data}
        return status_code, headers, body, elapsed
    except Exception as e:
        print(f"[FAIL] 서버 연결 실패 ({url}): {e}")
        print("  -> FastAPI 서버가 실행 중인지 확인하세요: python app.py")
        sys.exit(1)


def http_post_json(url: str, payload: dict):
    """POST JSON 요청을 보내고 (status_code, headers, body_json, elapsed_time)을 반환한다."""
    data = json.dumps(payload).encode("utf-8")
    start_t = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        elapsed = time.perf_counter() - start_t
        status_code = res.getcode()
        headers = dict(res.info())
        body = json.loads(res.read().decode("utf-8"))
        return status_code, headers, body, elapsed


def test_api_spec_protocol_and_headers():
    """명세서 (2) API 기본 명세: HTTP 200, UTF-8, CORS 검증"""
    print("\n" + "=" * 60)
    print("[TEST 1] 명세서 (2) API 기본 명세 (HTTP 200, UTF-8 Encoding, CORS 검증)")
    print("=" * 60)

    status_code, headers, body, elapsed = http_get(f"{BASE_URL}/current_region", params={"region_id": "GIMHAE"})
    
    # 1. HTTP 200 OK 정상 응답 확인
    assert status_code == 200, f"Expected HTTP 200 OK, got {status_code}"
    print("  ✓ 정상 응답: HTTP 200 OK 확인")

    # 2. Content-Type 및 UTF-8 Encoding 확인
    content_type = headers.get("content-type", "").lower()
    assert "application/json" in content_type, f"Expected application/json, got {content_type}"
    assert "utf-8" in content_type or "charset=utf-8" in content_type, f"Expected utf-8 charset, got {content_type}"
    print(f"  ✓ 인코딩 명세: UTF-8 JSON 확인 ({content_type})")

    # 3. CORS 헤더 확인 (전시 웹 프론트엔드 연계 대비)
    cors_header = headers.get("access-control-allow-origin") or headers.get("Access-Control-Allow-Origin")
    assert cors_header == "*", f"Expected Access-Control-Allow-Origin: *, got {cors_header}"
    print(f"  ✓ 전시 연계 헤더: CORS allow-origin=* 확인")


def test_missing_and_invalid_region_id():
    """명세서 (6) 오류 Response 권장: region_id 누락 및 잘못된 지역 요청 검증"""
    print("\n" + "=" * 60)
    print("[TEST 2] 명세서 (6) 오류 Response 권장 검증 (region_id 누락 및 타 지역 요청)")
    print("=" * 60)

    # 1. region_id 누락 요청 (/current_region)
    print("  -> [케이스 A] region_id 쿼리 파라미터 완전 누락")
    status_code, _, data, _ = http_get(f"{BASE_URL}/current_region")
    assert status_code == 200, f"Expected 200, got {status_code}"
    assert "region_id" in data and "message" in data, f"Required error fields missing in {data}"
    assert data["region_id"] == "GIMHAE", f"Expected region_id 'GIMHAE', got {data.get('region_id')}"
    assert data["message"] == "Requested region_id was not found.", f"Unexpected message: {data.get('message')}"
    print("     ✓ 통과: {\"region_id\": \"GIMHAE\", \"message\": \"Requested region_id was not found.\"}")

    # 2. region_id 빈 문자열 (/current_region?region_id=)
    print("  -> [케이스 B] region_id 파라미터가 빈 문자열인 경우")
    status_code, _, data, _ = http_get(f"{BASE_URL}/current_region", params={"region_id": ""})
    assert status_code == 200, f"Expected 200, got {status_code}"
    assert data.get("message") == "Requested region_id was not found."
    print("     ✓ 통과: 빈 파라미터 오류 처리 확인")

    # 3. 명세서 예시에 나온 타 지역 ID (/current_region?region_id=DAMYANG)
    print("  -> [케이스 C] 타 지역 요청 (명세서 예시: DAMYANG)")
    status_code, _, data, _ = http_get(f"{BASE_URL}/current_region", params={"region_id": "DAMYANG"})
    assert status_code == 200, f"Expected 200, got {status_code}"
    assert data.get("region_id") == "GIMHAE", f"Expected region_id 'GIMHAE', got {data.get('region_id')}"
    assert data.get("message") == "Requested region_id was not found.", f"Unexpected message: {data.get('message')}"
    print("     ✓ 통과: 타 지역 요청 시 오류 권장 규격 일치 확인")


def test_valid_current_region_spec(csv_places: dict):
    """명세서 (4) 응답 형식 및 (5) 필수 Response Field 검증"""
    print("\n" + "=" * 60)
    print("[TEST 3] 명세서 (4), (5) 필수 Response Field 및 CSV 기준 데이터 일치 검증")
    print("=" * 60)

    status_code, _, data, elapsed = http_get(f"{BASE_URL}/current_region", params={"region_id": "GIMHAE"})
    assert status_code == 200, f"Expected 200, got {status_code}"
    print(f"  -> 응답 수신 완료 (소요 시간: {elapsed:.3f}초)")

    # 1. Root 필드 검증
    expected_root_keys = {"region_id", "recommendations"}
    assert set(data.keys()) == expected_root_keys, (
        f"Root fields mismatch! Expected exactly {expected_root_keys}, got {set(data.keys())}"
    )
    assert data["region_id"] == "GIMHAE", f"Expected region_id 'GIMHAE', got {data['region_id']}"
    assert isinstance(data["recommendations"], list), "recommendations should be a JSON Array"
    print("  ✓ Root 필드 규격 일치: region_id (String), recommendations (Array)")

    # 2. 추천 POI 개수 검증 (최소 5개 이상)
    recommendations = data["recommendations"]
    assert len(recommendations) >= 5, f"Expected at least 5 recommendations, got {len(recommendations)}"
    print(f"  ✓ 추천 장소 개수: {len(recommendations)}개 (명세서 '최소 5개 이상' 요건 충족)")

    # 3. 중복 추천 여부 검증
    seen_ids = set()
    allowed_poi_keys = {"place_name", "place_id", "recommend_reason"}

    for idx, item in enumerate(recommendations, 1):
        assert isinstance(item, dict), f"Item #{idx} is not an object"
        item_keys = set(item.keys())
        assert item_keys == allowed_poi_keys, (
            f"Item #{idx} fields mismatch! Expected exactly {allowed_poi_keys}, got {item_keys}"
        )

        pid = item["place_id"]
        pname = item["place_name"]
        reason = item["recommend_reason"]

        # 중복 여부 확인
        assert pid not in seen_ids, f"Duplicate place_id found: '{pid}' in recommendation #{idx}"
        seen_ids.add(pid)

        # 4. place_id가 지역 기준 데이터(CSV)에 존재하는지 검증
        assert pid in csv_places, f"Item #{idx} place_id '{pid}' is not in CSV dataset!"

        # 5. place_name이 지역 기준 데이터(CSV)와 100% 일치하는지 검증
        expected_pname = csv_places[pid]
        assert pname == expected_pname, (
            f"Item #{idx} place_name mismatch! ID: {pid}, Expected: '{expected_pname}', Got: '{pname}'"
        )

        # 6. recommend_reason 문장 규칙 검증 (2~5문장, 종결부호)
        sentences = [s.strip() for s in re.split(r"[.!?]", reason) if s.strip()]
        assert 2 <= len(sentences) <= 5, (
            f"Item #{idx} sentence count violation: {len(sentences)} sentences found (must be 2~5)"
        )
        assert reason[-1] in (".", "!", "?"), (
            f"Item #{idx} reason must end with terminal punctuation: '{reason}'"
        )

        print(f"     [#{idx} OK] {pid} | {pname} | 일치 확인 (문장 수: {len(sentences)})")
        print(f"          추천 사유: {reason}")

    print("  ✓ 모든 추천 항목이 명세서 필수 필드 및 CSV 데이터와 100% 일치합니다.")


def test_case_insensitivity_and_caching(csv_places: dict):
    """현장 안정성 검증: 대소문자 유연성 및 캐시 응답 속도 (타임아웃 방지)"""
    print("\n" + "=" * 60)
    print("[TEST 4] 현장 안정성 검증 (대소문자 유연성 및 캐시 응답 지연시간)")
    print("=" * 60)

    # 1. 소문자 'gimhae' 요청 테스트
    print("  -> [케이스 A] region_id 소문자 입력 ('gimhae')")
    status_code, _, data, elapsed_case = http_get(f"{BASE_URL}/current_region", params={"region_id": "gimhae"})
    assert status_code == 200
    assert data.get("region_id") == "GIMHAE"
    assert "recommendations" in data
    assert len(data["recommendations"]) >= 5
    print(f"     ✓ 통과: 소문자 'gimhae' 정상 처리 완료 (응답 속도: {elapsed_case:.3f}초)")

    # 2. 취향 변경 (POST /preference) 및 캐싱 검증
    print("\n  -> [케이스 B] 취향 변경(POST /preference) 후 초고속 캐시 응답 검증")
    pref_text = "조용한 숲 산책로와 자연 경관"
    _, _, pref_res, pref_elapsed = http_post_json(f"{BASE_URL}/preference", {"preference": pref_text})
    assert pref_res.get("preference") == pref_text
    print(f"     ✓ POST /preference 취향 저장 및 캐시 사전생성 완료 (소요 시간: {pref_elapsed:.3f}초)")

    # 3. /current_region 재호출 시 캐시 즉각 응답 시간 확인 (목표: < 0.1초)
    status_code, _, cache_data, cache_elapsed = http_get(f"{BASE_URL}/current_region", params={"region_id": "GIMHAE"})
    assert status_code == 200
    assert len(cache_data["recommendations"]) >= 5
    print(f"     ✓ GET /current_region 캐시 즉각 응답 성공! (응답 시간: {cache_elapsed * 1000:.1f}ms)")
    
    # 응답 시간이 1초 이내인지 검증 (전시 시스템 타임아웃 절대 방지)
    assert cache_elapsed < 1.0, f"Cache response took too long: {cache_elapsed:.3f}s"
    print("     ✓ 전시 부스 HTTP 타임아웃 방지 기준(< 1.0초) 완벽 충족")


def main():
    print("============================================================")
    print(" Re:Local 연계 REST API 명세서 전체 요구 조건 자동 검증")
    print(f" 대상 서버: {BASE_URL}")
    print("============================================================")

    csv_places = load_csv_places()
    test_api_spec_protocol_and_headers()
    test_missing_and_invalid_region_id()
    test_valid_current_region_spec(csv_places)
    test_case_insensitivity_and_caching(csv_places)

    print("\n" + "=" * 60)
    print(" 🎉 [SUCCESS] 명세서의 모든 요구 조건과 현장 안정성 검증을 100% 통과했습니다!")
    print("=" * 60)


if __name__ == "__main__":
    main()
