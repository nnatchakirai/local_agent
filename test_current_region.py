#!/usr/bin/env python3
"""
/current_region 엔드포인트 명세서 준수 및 place_id/place_name 일치 검증 테스트 스크립트
(내장 urllib 모듈 사용으로 추가 패키지 설치 불필요)
"""

import sys
import re
import json
import urllib.request
import urllib.parse
import urllib.error
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"
CSV_PATH = "여행지_20_김해_수정본.csv"


def load_csv_places():
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


def http_get(url, params=None):
    if params:
        query_string = urllib.parse.urlencode(params)
        url = f"{url}?{query_string}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as res:
        status_code = res.getcode()
        body = json.loads(res.read().decode("utf-8"))
        return status_code, body


def http_post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST"
    )
    with urllib.request.urlopen(req) as res:
        status_code = res.getcode()
        body = json.loads(res.read().decode("utf-8"))
        return status_code, body


def test_missing_region_id():
    """region_id 누락 시 올바른 에러 메시지 반환 여부 검증"""
    print("\n[TEST 1] region_id 파라미터 누락 검증")
    status_code, data = http_get(f"{BASE_URL}/current_region")
    assert status_code == 200, f"Expected 200, got {status_code}"
    assert data.get("region_id") == "GIMHAE", f"Expected region_id 'GIMHAE', got {data.get('region_id')}"
    assert data.get("message") == "Requested region_id was not found.", f"Unexpected message: {data.get('message')}"
    print("  -> 통과: {\"region_id\":\"GIMHAE\",\"message\":\"Requested region_id was not found.\"}")


def test_invalid_region_id():
    """region_id가 GIMHAE가 아닐 때 올바른 에러 메시지 반환 여부 검증"""
    print("\n[TEST 2] region_id가 GIMHAE가 아닌 경우 검증")
    status_code, data = http_get(f"{BASE_URL}/current_region", params={"region_id": "SEOUL"})
    assert status_code == 200, f"Expected 200, got {status_code}"
    assert data.get("region_id") == "GIMHAE", f"Expected region_id 'GIMHAE', got {data.get('region_id')}"
    assert data.get("message") == "Requested region_id was not found.", f"Unexpected message: {data.get('message')}"
    print("  -> 통과: {\"region_id\":\"GIMHAE\",\"message\":\"Requested region_id was not found.\"}")


def test_save_preference():
    """POST /preference를 통해 사용자 취향 저장 검증"""
    print("\n[TEST 3] POST /preference 사용자 취향 저장")
    pref_text = "조용한 가야 역사 유적지와 숲 산책로"
    status_code, data = http_post_json(f"{BASE_URL}/preference", {"preference": pref_text})
    assert status_code == 200, f"Expected 200, got {status_code}"
    assert data.get("preference") == pref_text, f"Preference mismatch: {data}"
    print(f"  -> 통과: preference 정상 저장 완료 ('{pref_text}')")


def test_valid_current_region(csv_places: dict):
    """정상 호출 시 응답 규격 및 CSV 데이터 일치 여부 검증"""
    print("\n[TEST 4] GET /current_region?region_id=GIMHAE 정상 추천 및 CSV 일치 검증 (LLM 추론 중...)")
    status_code, data = http_get(f"{BASE_URL}/current_region", params={"region_id": "GIMHAE"})
    assert status_code == 200, f"Expected 200, got {status_code}"
    
    assert "region_id" in data, "Root field 'region_id' missing"
    assert data["region_id"] == "GIMHAE", f"Expected region_id 'GIMHAE', got {data['region_id']}"
    assert "recommendations" in data, "Root field 'recommendations' missing"

    recommendations = data["recommendations"]
    assert isinstance(recommendations, list), "recommendations should be a list"
    assert len(recommendations) >= 5, f"Expected at least 5 recommendations, got {len(recommendations)}"
    print(f"  -> 추천 장소 개수: {len(recommendations)}개 (최소 5개 요건 충족)")

    allowed_keys = {"place_name", "place_id", "recommend_reason"}

    for idx, item in enumerate(recommendations, 1):
        item_keys = set(item.keys())
        assert item_keys == allowed_keys, (
            f"Item #{idx} keys mismatch! Expected exactly {allowed_keys}, got {item_keys}"
        )

        pid = item["place_id"]
        pname = item["place_name"]
        reason = item["recommend_reason"]

        # 1. place_id 존재 검증
        assert pid in csv_places, f"Item #{idx} place_id '{pid}' is not in CSV!"

        # 2. place_name 100% 일치 검증 (CSV strip 값과)
        expected_pname = csv_places[pid]
        assert pname == expected_pname, (
            f"Item #{idx} place_name mismatch! ID: {pid}, Expected: '{expected_pname}', Got: '{pname}'"
        )

        # 3. recommend_reason 문장 규칙 검증 (2~5문장, 종결부호)
        sentences = [s.strip() for s in re.split(r"[.!?]", reason) if s.strip()]
        assert 2 <= len(sentences) <= 5, (
            f"Item #{idx} sentence count violation: {len(sentences)} sentences found (must be 2~5)"
        )
        assert reason[-1] in (".", "!", "?"), f"Item #{idx} reason must end with terminal punctuation: {reason}"

        print(f"  [#{idx} OK] {pid}: {pname} | 일치 확인 완료 (문장 수: {len(sentences)})")

    print("\n>>> 모든 항목이 명세서 및 CSV 데이터와 100% 일치합니다! <<<")


def main():
    csv_places = load_csv_places()
    test_missing_region_id()
    test_invalid_region_id()
    test_save_preference()
    test_valid_current_region(csv_places)
    print("\n[SUCCESS] 모든 검증 테스트를 성공적으로 통과했습니다.")


if __name__ == "__main__":
    main()
