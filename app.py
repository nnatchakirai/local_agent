"""
AIM Challenge 2026 - 김해(GIMHAE) 지역추천 REST API
정량평가 가이드 기준 보완판

기존 app.py 대비 변경점
1) <2문장 보완 문구를 evidence_text 기반 근거 문장으로 교체
   - 기존: "{장소}에서 특별한 추억과 즐거운 경험을 만들어보세요." (근거 없음 → faith_ratio 감점 위험)
   - 변경: evidence_text_2(주관) → evidence_text_1(객관) → summary(테마) 순으로 실제 데이터 문장을 가져와 보충
   - 그래도 2문장이 안 되면(데이터 자체가 부실) 해당 장소는 이번 후보에서 제외하고 재요청으로 대체
2) 최종 추천 수가 5곳 미만이면 LLM에 부족분만큼 재요청(backfill)하여 5곳을 채움
   - 이미 선택된 장소는 제외 목록으로 프롬프트에 명시 + 서버단 seen_ids로 이중 방어
3) matching_tags를 2~3개로 클램핑(중복 제거, 부족 시 보충, 초과 시 절단)
4) 시스템 프롬프트에 "문장 사이 구두점 누락 시 0점 처리" 경고를 명시적으로 추가
"""

import json
import re
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI()

TARGET_COUNT = 5     # 제출 규격상 요구되는 추천 개수
MAX_ATTEMPTS = 3     # 최초 1회 + 부족분 보충 최대 2회

# 김해 관광지 데이터셋 로드
CSV_PATH = "여행지_20_김해_수정본.csv"
try:
    df_places = pd.read_csv(CSV_PATH)

    # place_id 매핑용 사전 준비 (100% 정확한 place_id, place_name 매칭 보장)
    VALID_PLACES = {
        str(row['place_id']).strip(): str(row['place_name']).strip()
        for _, row in df_places.iterrows()
    }

    # 평가 가이드의 근거성(grounding) 3축(객관/주관/테마)에 대응하는
    # evidence 원문을 장소별로 미리 인덱싱 (문장 보완 시 사용)
    EVIDENCE_MAP = {
        str(row['place_id']).strip(): {
            "objective": str(row.get('evidence_text_1', '') or '').strip(),
            "subjective": str(row.get('evidence_text_2', '') or '').strip(),
            "theme": str(row.get('summary', '') or '').strip(),
        }
        for _, row in df_places.iterrows()
    }
    print("로컬 관광지 데이터 로드 성공!")
except Exception as e:
    df_places = None
    VALID_PLACES = {}
    EVIDENCE_MAP = {}
    print(f"CSV 로드 실패: {e}. 'app.py'와 같은 폴더에 csv 파일이 있는지 확인하세요.")


class UserRequest(BaseModel):
    preference: str  # 사용자의 취향 입력


def count_valid_sentences(text: str) -> int:
    """
    평가 가이드 규칙에 맞춰 유효 문장 수를 산출하는 함수
    1. 구분자: 마침표(.), 느낌표(!), 물음표(?)
    2. 쉼표(,)는 구분자로 인정하지 않음
    3. 5자 이하의 짧은 조각은 문장 수에서 제외
    """
    raw_sentences = re.split(r'[.!?]', text)
    valid_sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 5]
    return len(valid_sentences)


def get_grounded_filler(place_id: str, current_reason: str) -> str:
    """
    recommend_reason이 2문장 미만일 때, 근거 없는 일반 문구 대신
    evidence_text_2(주관) → evidence_text_1(객관) → summary(테마) 순으로
    실제 데이터에 있는 문장을 가져와 보충한다.

    이렇게 해야 faith_ratio(설명 충실도) 계산에서 보충 문장도
    "지지(supported)"로 판정될 가능성이 높다. 기존처럼 일반적인
    응원 문구를 넣으면 그 문장은 evidence와 유사도가 낮아
    오히려 충실도 점수를 깎을 수 있다.
    """
    evidence = EVIDENCE_MAP.get(place_id, {})
    candidates = [
        evidence.get("subjective", ""),
        evidence.get("objective", ""),
        evidence.get("theme", ""),
    ]

    for cand in candidates:
        cand = str(cand).strip()
        if not cand:
            continue
        # 이미 reason에 거의 그대로 들어있는 문장은 중복이므로 건너뜀
        if len(cand) >= 15 and cand[:15] in current_reason:
            continue
        if cand[-1] not in ['.', '!', '?']:
            cand += "."
        if len(cand) > 5:
            return " " + cand

    return ""  # 세 축 모두 비어있는 예외적인 경우


def clamp_tags(tags) -> list:
    """
    matching_tags를 가이드 기준(2~3개)에 맞게 정리한다.
    - 리스트가 아니거나 빈 값 제거
    - 중복 제거(순서 유지)
    - 3개 초과 시 절단, 2개 미만 시 보충
    """
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip() for t in tags if str(t).strip()]

    seen = set()
    uniq = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    tags = uniq

    if len(tags) > 3:
        tags = tags[:3]

    if len(tags) < 2:
        for fallback in ["김해여행", "김해관광", "추천여행지"]:
            if fallback not in tags:
                tags.append(fallback)
            if len(tags) >= 2:
                break

    return tags


def validate_and_fix_response(result_data: list, already_seen_ids: set, limit: int) -> list:
    """
    LLM 생성이 규정을 위반했을 때 0점 처리를 방지하기 위한 후처리 가드레일

    already_seen_ids: 이전 시도(원 요청 + backfill 재요청)에서 이미 채택된 place_id
                       → 여러 번의 LLM 호출에 걸친 중복도 방지
    limit: 이번 호출에서 최대 몇 개까지 채택할지 (필요한 만큼만 채택)
    """
    validated_list = []

    for item in result_data:
        p_id = str(item.get("place_id", "")).strip()
        reason = str(item.get("recommend_reason", "")).strip()
        tags = item.get("matching_tags", [])
        img_prompt = str(item.get("image_prompt", "")).strip()

        # 1. 중복 place_id 검증 (이번 호출 내 + 이전 호출 누적 모두 확인)
        if p_id in already_seen_ids:
            continue

        # 2. place_id 원본 데이터 존재 여부 100% 일치 검증
        if p_id not in VALID_PLACES:
            continue
        # 원본 데이터 기준 장소명으로 강제 동기화 (오기입 0점 방지)
        p_name = VALID_PLACES[p_id]

        # 3. 문장 부호 보완 (끝에 문장부호 없으면 마침표 추가)
        if reason and reason[-1] not in ['.', '!', '?']:
            reason += "."

        # 4. 문장 수 검증 (2~5개 조건) — 근거 기반 보완으로 개선
        sent_count = count_valid_sentences(reason)
        if sent_count < 2:
            filler = get_grounded_filler(p_id, reason)
            if filler:
                reason += filler
                sent_count = count_valid_sentences(reason)
            if sent_count < 2:
                # evidence로 보완해도 2문장이 안 되면(데이터 자체가 부실)
                # 0점 리스크를 감수하기보다 이 장소는 제외하고 backfill로 대체
                continue

        # 5. matching_tags 2~3개로 정리
        tags = clamp_tags(tags)

        # 6. image_prompt 미제공 시 기본값 보완
        if not img_prompt:
            img_prompt = f"A realistic high-quality photo of {p_name} in Gimhae, beautifully styled travel destination."

        already_seen_ids.add(p_id)
        validated_list.append({
            "place_id": p_id,
            "place_name": p_name,
            "recommend_reason": reason,
            "matching_tags": tags,
            "image_prompt": img_prompt
        })

        if len(validated_list) == limit:
            break

    return validated_list


def build_context_data(df: pd.DataFrame, exclude_ids: set) -> str:
    """관광지 컨텍스트 텍스트 생성. 이미 채택된 장소는 재추천 후보에서 제외."""
    context_data = ""
    for _, row in df.iterrows():
        pid = str(row['place_id']).strip()
        if pid in exclude_ids:
            continue

        extra_evidence = " ".join(
            str(row.get(f"evidence_text_{i}", "") or "") for i in range(3, 6)
        ).strip()

        context_data += (
            f"장소ID: {row['place_id']}\n"
            f"장소명: {row['place_name']}\n"
            f"객관적사실: {row['evidence_text_1']}\n"
            f"주관적사실: {row['evidence_text_2']}\n"
            f"테마: {row['summary']}\n"
        )
        if extra_evidence:
            context_data += f"추가정보: {extra_evidence}\n"
        context_data += "-------------------\n"
    return context_data


def build_system_instruction(count_needed: int, exclude_names: list) -> str:
    exclude_note = ""
    if exclude_names:
        exclude_note = (
            "\n[중복 방지]: 다음 장소는 이미 추천 목록에 포함되었으니 절대 다시 추천하지 마라: "
            + ", ".join(exclude_names) + "\n"
        )

    return (
        "너는 김해시 전문 여행 가이드 AI 에이전트야.\n"
        "제공된 관광지 데이터와 사용자의 취향을 분석해서 가장 적합한 곳을 정확히 "
        f"{count_needed}곳 선정하고,\n"
        "반드시 지정된 JSON 배열 형식으로만 답변해줘. 마크다운 기호(```json 등)나 부가 설명은 절대 하지마."
        f"{exclude_note}\n"
        "[엄격한 정량평가 및 제출 규칙]:\n"
        "1. 각 객체는 'place_id', 'place_name', 'recommend_reason', 'matching_tags', 'image_prompt' 5개 필드를 반드시 포함해야 한다.\n"
        "2. 'place_id'와 'place_name'은 제공된 데이터의 값과 100% 동일하게 적어야 한다.\n"
        "3. 'recommend_reason'은 반드시 2문장 이상 5문장 이하이어야 하며, 각 문장은 5자 이상의 완성된 문장이어야 한다.\n"
        "4. 모든 문장의 끝에는 반드시 마침표(.), 느낌표(!), 물음표(?) 중 하나를 찍어야 한다. "
        "문장과 문장 사이에도 반드시 구두점을 찍어라. 구두점 없이 두 문장을 이어 쓰면 "
        "전체가 1문장으로 인식되어 해당 장소가 0점 처리된다. (쉼표는 문장 구분 안 됨)\n"
        "5. 데이터에 없는 사실(주차비, 무료입장, 지어낸 축제 등)은 절대 추가하지 마라.\n"
        "6. 'matching_tags'는 해당 장소의 주요 키워드 2~3개의 리스트 형태로 작성하라.\n"
        "7. 'image_prompt'는 해당 관광지의 시각적 특징을 담은 영문 이미지 생성 프롬프트로 작성하라.\n\n"
        "출력 예시:\n"
        "[\n"
        "  {\n"
        "    \"place_id\": \"GIMHAE_01\",\n"
        "    \"place_name\": \"김해 수로왕릉\",\n"
        "    \"recommend_reason\": \"수로왕릉은 가야의 역사를 느낄 수 있는 대표 유적지입니다. "
        "한적하게 산책하며 역사 공부를 하기에 아주 좋습니다.\",\n"
        "    \"matching_tags\": [\"역사탐방\", \"가야유적\", \"산책로\"],\n"
        "    \"image_prompt\": \"A peaceful photorealistic view of King Suro Tomb in Gimhae, "
        "traditional Korean royal tomb surrounded by green trees.\"\n"
        "  }\n"
        "]"
    )


def call_llm(user_pref: str, context_data: str, count_needed: int, exclude_names: list) -> list:
    """LLM 호출 + 마크다운 코드블록 제거 + JSON 파싱까지 담당."""
    system_instruction = build_system_instruction(count_needed, exclude_names)
    user_message = f"사용자 취향: {user_pref}\n\n김해 관광지 데이터:\n{context_data}"

    response = ollama.generate(
        model="qwen2.5:7b",
        system=system_instruction,
        prompt=user_message,
        options={"temperature": 0.1, "num_ctx": 8192}
    )

    reply_text = response["response"].strip()

    if reply_text.startswith("```"):
        reply_text = re.sub(r"^```(?:json)?\s*", "", reply_text)
        reply_text = re.sub(r"\s*```$", "", reply_text)

    return json.loads(reply_text)


@app.post("/recommend")
async def recommend_places(req: UserRequest):
    if df_places is None or len(df_places) == 0:
        raise HTTPException(status_code=500, detail="관광지 데이터 파일이 존재하지 않습니다.")

    user_pref = req.preference
    validated_list: list = []
    seen_ids: set = set()

    for attempt in range(MAX_ATTEMPTS):
        needed = TARGET_COUNT - len(validated_list)
        if needed <= 0:
            break

        exclude_names = [VALID_PLACES[pid] for pid in seen_ids if pid in VALID_PLACES]
        context_data = build_context_data(df_places, exclude_ids=seen_ids)

        try:
            result_data = call_llm(user_pref, context_data, needed, exclude_names)
        except Exception as e:
            print(f"[시도 {attempt + 1}/{MAX_ATTEMPTS}] LLM 호출/파싱 실패: {e}")
            continue

        new_items = validate_and_fix_response(result_data, seen_ids, needed)
        validated_list.extend(new_items)

        if len(validated_list) >= TARGET_COUNT:
            break

    if not validated_list:
        raise HTTPException(status_code=500, detail="유효한 추천 결과를 생성하지 못했습니다.")

    if len(validated_list) < TARGET_COUNT:
        print(
            f"⚠ 최종 추천 수가 {len(validated_list)}곳으로 {TARGET_COUNT}곳에 미달합니다. "
            f"유효성(validity_ratio) 평가에 영향이 있을 수 있으니 데이터/프롬프트 점검이 필요합니다."
        )

    return validated_list[:TARGET_COUNT]
