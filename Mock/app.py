from devlog import logs
from fastapi.responses import JSONResponse
import json
import os
import re
import time
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import ollama
from devlog import dev_log

class UTF8JSONResponse(JSONResponse):
    media_type = "application/json; charset=utf-8"

app = FastAPI(default_response_class=UTF8JSONResponse)

# 전시/실감콘텐츠 및 타 시스템 웹 프론트엔드 연계를 위한 CORS 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 개발용: 어떤 API(데코레이터/함수)가 실행됐는지 한글 진행 로그로 남긴다.
_LOG_SKIP_PREFIXES = ("/api/dev-log", "/static")


@app.middleware("http")
async def log_api_call(request, call_next):
    path = request.url.path
    if path == "/" or path.startswith(_LOG_SKIP_PREFIXES):
        return await call_next(request)

    start = time.time()
    dev_log(f"API 요청 수신: {request.method} {path}")
    response = await call_next(request)

    # 라우팅 후에만 scope["route"]가 채워진다
    route = request.scope.get("route")
    endpoint = getattr(route, "endpoint", None)
    if endpoint is not None:
        owner = "app" if endpoint.__module__ in ("app", "__main__") else "router"
        method = request.method.lower()
        dev_log(
            f'API 실행 완료: @{owner}.{method}("{route.path}") → {endpoint.__name__}() '
            f"[{response.status_code}, {int((time.time() - start) * 1000)}ms]"
        )
    return response

# =========================================================
# 기본 설정
# ========================================= ===============

TARGET_COUNT = 5        # 제출 규격상 요구되는 추천 개수
MAX_ATTEMPTS = 3        # LLM 재요청(backfill) 최대 횟수
MIN_SENTENCES = 2       # 추천 이유 최소 문장 수
MAX_SENTENCES = 5       # 추천 이유 최대 문장 수 (초과분은 평가에 반영 안 됨)
MIN_TAGS = 2
MAX_TAGS = 3
THEME_TAG_MAX_LEN = 15  # summary가 이 길이를 넘으면 "태그"로 보지 않고 폴백 사용

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "여행지_20_김해_수정본.csv")
OLLAMA_MODEL = "qwen2.5:7b"

TAG_FALLBACK_POOL = ["김해여행", "관광명소", "추천명소"]

TERMINAL_PUNCT = (".", "!", "?")


# =========================================================
# CSV 데이터 로드
# =========================================================

try:
    df_places = pd.read_csv(CSV_PATH)
except Exception as e:
    df_places = None
    print(f"[ERROR] CSV 로드 실패: {e}")


# =========================================================
# 장소 데이터 구성
# =========================================================

VALID_PLACES = {}   # place_id -> place_name
EVIDENCE_MAP = {}    # place_id -> {"objective", "subjective", "evidence_3~5", "theme"}

if df_places is not None:
    for _, row in df_places.iterrows():
        place_id = str(row.get("place_id", "")).strip()
        place_name = str(row.get("place_name", "")).strip()

        if not place_id or not place_name:
            continue

        VALID_PLACES[place_id] = place_name

        EVIDENCE_MAP[place_id] = {
            "objective": str(row.get("evidence_text_1", "") or "").strip(),
            "subjective": str(row.get("evidence_text_2", "") or "").strip(),
            "evidence_3": str(row.get("evidence_text_3", "") or "").strip(),
            "evidence_4": str(row.get("evidence_text_4", "") or "").strip(),
            "evidence_5": str(row.get("evidence_text_5", "") or "").strip(),
            "theme": str(row.get("summary", "") or "").strip(),
        }

    print(f"로컬 관광지 데이터 로드 성공: {len(VALID_PLACES)}개")


# =========================================================
# Request Model
# =========================================================

class UserRequest(BaseModel):
    preference: str


# =========================================================
# 공통 유틸
# =========================================================

def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def default_image_prompt(place_name: str) -> str:
    return f"A realistic high-quality travel photo of {place_name} in Gimhae."


def theme_as_tag_candidate(theme: str) -> str:
    """
    summary(테마) 필드가 짧은 키워드 형태일 때만 태그 후보로 사용한다.
    문장형으로 길면(THEME_TAG_MAX_LEN 초과) 태그로 쓰기 부적절하므로
    빈 문자열을 반환해 표준 폴백 태그가 대신 쓰이게 한다.
    """
    theme = clean_text(theme)
    if not theme or len(theme) > THEME_TAG_MAX_LEN:
        return ""
    return theme


# =========================================================
# 문장 분리 / 개수 세기
# =========================================================

def split_sentences(text: str) -> list:
    """
    AIM 평가 기준에 맞춘 문장 분리.
    - 구분자: '.', '!', '?' (쉼표는 구분자 아님)
    - 5자 이하 조각은 유효 문장에서 제외
    - 공백 유무와 무관하게 구두점만으로 판정 (평가 가이드 원문 그대로 반영)
    """
    text = clean_text(text)
    if not text:
        return []

    raw_sentences = re.split(r"[.!?]", text)

    valid_sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if len(sentence) <= 5:
            continue
        valid_sentences.append(sentence)

    return valid_sentences


def count_valid_sentences(text: str) -> int:
    return len(split_sentences(text))


def fix_missing_internal_punctuation(text: str) -> str:
    """
    LLM이 한국어 문장 사이의 마침표를 누락시키는 경우 일부 보정한다.
    예: "역사적인 장소입니다 수로왕릉은..." -> "역사적인 장소입니다. 수로왕릉은..."
    """
    text = clean_text(text)
    if not text:
        return ""

    pattern = (
        r"([가-힣]{2,}(?:습니다|니다|해요|이에요|예요|이죠|군요|했죠|있죠|없죠|합니다|됩니다))"
        r"\s+(?=[가-힣A-Za-z0-9])"
    )
    text = re.sub(pattern, r"\1. ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text and text[-1] not in TERMINAL_PUNCT:
        text += "."

    return text


def limit_to_five_sentences(text: str) -> str:
    """
    추천 이유를 최대 MAX_SENTENCES 문장으로 제한한다.

    [수정] 기존에는 "(?<=[.!?])\\s+" 로 분리해서 구두점 뒤 공백이 없으면
    문장이 안 쪼개지는 문제가 있었다. re.findall로 "구두점까지 포함한 덩어리"를
    직접 찾아내는 방식으로 바꿔 공백 유무와 무관하게 동작하도록 했다.
    """
    text = clean_text(text)
    if not text:
        return ""

    text = fix_missing_internal_punctuation(text)
    if not text:
        return ""

    raw_sentences = re.findall(r"[^.!?]+[.!?]+", text)
    if not raw_sentences:
        raw_sentences = [text if text[-1] in TERMINAL_PUNCT else text + "."]

    valid_sentences = []
    for sentence in raw_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if sentence[-1] not in TERMINAL_PUNCT:
            sentence += "."

        content = re.sub(r"[.!?]+$", "", sentence).strip()
        if len(content) <= 5:
            continue

        valid_sentences.append(sentence)
        if len(valid_sentences) >= MAX_SENTENCES:
            break

    return " ".join(valid_sentences)


# =========================================================
# Evidence 후보
# =========================================================

def get_evidence_candidates(place_id: str) -> list:
    """
    해당 장소의 모든 evidence를 우선순위대로 가져온다.
    1. evidence_text_1(객관) 2. evidence_text_2(주관)
    3~5. evidence_text_3~5  6. summary(테마)
    """
    evidence = EVIDENCE_MAP.get(place_id, {})
    candidates = [
        evidence.get("objective", ""),
        evidence.get("subjective", ""),
        evidence.get("evidence_3", ""),
        evidence.get("evidence_4", ""),
        evidence.get("evidence_5", ""),
        evidence.get("theme", ""),
    ]
    return [clean_text(c) for c in candidates if clean_text(c)]


def get_grounded_filler(place_id: str, current_reason: str) -> str:
    """
    추천 이유가 MIN_SENTENCES 미만일 경우 실제 CSV evidence로 보완한다.
    임의의 홍보 문구는 만들지 않는다.
    """
    current_reason = clean_text(current_reason)
    candidates = get_evidence_candidates(place_id)

    for candidate in candidates:
        if len(candidate) <= 5:
            continue

        if candidate[-1] not in TERMINAL_PUNCT:
            candidate += "."

        candidate_content = re.sub(r"[.!?]+$", "", candidate).strip()
        if candidate_content and candidate_content in current_reason:
            continue  # 이미 reason에 들어있는 근거는 건너뜀

        return " " + candidate

    return ""


# =========================================================
# Hard Backfill (LLM 없이 CSV만으로 최후 보충)
# =========================================================

def create_hard_backfill_item(place_id: str) -> dict:
    """
    LLM이 충분한 추천을 생성하지 못했을 때, CSV의 실제 evidence만 이어붙여
    최소 요건(2문장 이상)을 만족하는 추천 항목을 만든다.
    evidence가 부실해 2문장을 못 채우면 억지로 만들지 않고 빈 딕셔너리를 반환한다.
    """
    if place_id not in VALID_PLACES:
        return {}

    place_name = VALID_PLACES[place_id]
    candidates = get_evidence_candidates(place_id)

    parts = []
    for candidate in candidates:
        if len(candidate) <= 5:
            continue
        if candidate[-1] not in TERMINAL_PUNCT:
            candidate += "."
        if candidate in parts:
            continue
        parts.append(candidate)
        if len(parts) >= MAX_SENTENCES:
            break

    reason = " ".join(parts)
    reason = fix_missing_internal_punctuation(reason)
    reason = limit_to_five_sentences(reason)

    if count_valid_sentences(reason) < MIN_SENTENCES:
        return {}

    matching_tags = []
    theme_tag = theme_as_tag_candidate(EVIDENCE_MAP.get(place_id, {}).get("theme", ""))
    if theme_tag:
        matching_tags.append(theme_tag)
    for fallback in TAG_FALLBACK_POOL:
        if len(matching_tags) >= MIN_TAGS:
            break
        if fallback not in matching_tags:
            matching_tags.append(fallback)
    matching_tags = matching_tags[:MAX_TAGS]

    return {
        "place_id": place_id,
        "place_name": place_name,
        "recommend_reason": reason,
        "matching_tags": matching_tags,
        "image_prompt": default_image_prompt(place_name),
    }


def rank_places_by_preference(user_pref: str, candidate_ids: list) -> list:
    """
    하드 백필 시 CSV 순서(=항상 GIMHAE_001부터) 그대로 채우지 않도록,
    사용자 취향 문장과 장소 evidence 간의 단순 단어 겹침 개수로 우선순위를 매긴다.
    BGE-M3 임베딩 없이 쓸 수 있는 가벼운 휴리스틱이라 완전한 의미 매칭은 아니지만,
    최소한 "취향과 아예 무관한 장소부터 채워지는" 상황은 줄여준다.
    """
    pref_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", user_pref))
    if not pref_tokens:
        return candidate_ids

    def overlap_score(place_id: str) -> int:
        evidence = EVIDENCE_MAP.get(place_id, {})
        combined = " ".join([
            evidence.get("objective", ""),
            evidence.get("subjective", ""),
            evidence.get("theme", ""),
        ])
        combined_tokens = set(re.findall(r"[가-힣A-Za-z0-9]+", combined))
        return len(pref_tokens & combined_tokens)

    return sorted(candidate_ids, key=overlap_score, reverse=True)


# =========================================================
# LLM JSON 파싱
# =========================================================

def extract_json_array(text: str) -> list:
    """LLM이 JSON 앞뒤에 설명이나 markdown code fence를 붙이는 경우에도 배열을 추출한다."""
    text = clean_text(text)
    if not text:
        return []

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            recommendations = data.get("recommendations")
            if isinstance(recommendations, list):
                return recommendations
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    try:
        data = json.loads(text[start:end + 1])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        return []

    return []


# =========================================================
# 항목 검증 (공통 함수 — LLM 결과 검증과 최종 검증이 공유)
# =========================================================

def _build_validated_item(item: dict, seen_ids: set, allow_filler: bool) -> dict:
    """
    하나의 추천 항목을 AIM 요구사항에 맞게 검증/보정한다.
    통과하지 못하면 None을 반환한다.

    allow_filler=True  : LLM이 갓 생성한 결과 검증 시 (문장 부족하면 evidence로 보완)
    allow_filler=False : 이미 한 번 검증/보정된 항목의 최종 재검증 시 (추가 보완 없이 재확인만)
    """
    if not isinstance(item, dict):
        return None

    p_id = clean_text(item.get("place_id", ""))
    if p_id not in VALID_PLACES or p_id in seen_ids:
        return None
    p_name = VALID_PLACES[p_id]

    reason = clean_text(item.get("recommend_reason", ""))
    reason = fix_missing_internal_punctuation(reason)
    reason = limit_to_five_sentences(reason)

    if count_valid_sentences(reason) < MIN_SENTENCES and allow_filler:
        filler = get_grounded_filler(p_id, reason)
        if filler:
            reason = (reason + filler).strip()
            reason = fix_missing_internal_punctuation(reason)
            reason = limit_to_five_sentences(reason)

    if count_valid_sentences(reason) < MIN_SENTENCES:
        return None

    matching_tags = item.get("matching_tags", [])
    if not isinstance(matching_tags, list):
        matching_tags = []
    matching_tags = [clean_text(t) for t in matching_tags if clean_text(t)]
    matching_tags = list(dict.fromkeys(matching_tags))[:MAX_TAGS]

    if len(matching_tags) < MIN_TAGS:
        theme_tag = theme_as_tag_candidate(EVIDENCE_MAP.get(p_id, {}).get("theme", ""))
        if theme_tag and theme_tag not in matching_tags:
            matching_tags.append(theme_tag)

    for fallback in TAG_FALLBACK_POOL:
        if len(matching_tags) >= MIN_TAGS:
            break
        if fallback not in matching_tags:
            matching_tags.append(fallback)
    matching_tags = matching_tags[:MAX_TAGS]

    image_prompt = clean_text(item.get("image_prompt", ""))
    if not image_prompt:
        image_prompt = default_image_prompt(p_name)

    return {
        "place_id": p_id,
        "place_name": p_name,
        "recommend_reason": reason,
        "matching_tags": matching_tags,
        "image_prompt": image_prompt,
    }


def validate_and_fix_response(result_data: list, already_seen_ids: set, limit: int) -> list:
    """LLM 응답을 검증하고, 필요하면 evidence 기반 문장 보완(filler)까지 수행한다."""
    validated_list = []
    if not isinstance(result_data, list):
        return validated_list

    for item in result_data:
        validated_item = _build_validated_item(item, already_seen_ids, allow_filler=True)
        if validated_item is None:
            continue

        validated_list.append(validated_item)
        already_seen_ids.add(validated_item["place_id"])

        if len(validated_list) >= limit:
            break

    return validated_list


def final_validate(recommendations: list) -> list:
    """이미 검증/보정된 항목들을 마지막으로 한 번 더 확인한다 (추가 보완 없이 재확인만)."""
    final_list = []
    seen_ids = set()

    for item in recommendations:
        validated_item = _build_validated_item(item, seen_ids, allow_filler=False)
        if validated_item is None:
            continue

        final_list.append(validated_item)
        seen_ids.add(validated_item["place_id"])

        if len(final_list) >= TARGET_COUNT:
            break

    return final_list


# =========================================================
# Context / System Prompt
# =========================================================

def build_context_data(df: pd.DataFrame, exclude_ids: set) -> str:
    """LLM에 제공할 관광지 데이터. 이미 추천된 장소는 컨텍스트에서 아예 제외한다."""
    context_data = ""

    for _, row in df.iterrows():
        pid = clean_text(row.get("place_id", ""))
        if not pid or pid in exclude_ids:
            continue

        place_name = clean_text(row.get("place_name", ""))
        objective = clean_text(row.get("evidence_text_1", ""))
        subjective = clean_text(row.get("evidence_text_2", ""))
        evidence_3 = clean_text(row.get("evidence_text_3", ""))
        evidence_4 = clean_text(row.get("evidence_text_4", ""))
        evidence_5 = clean_text(row.get("evidence_text_5", ""))
        summary = clean_text(row.get("summary", ""))

        context_data += (
            f"장소ID: {pid}\n"
            f"장소명: {place_name}\n"
            f"객관적사실: {objective}\n"
            f"주관적사실: {subjective}\n"
            f"추가근거1: {evidence_3}\n"
            f"추가근거2: {evidence_4}\n"
            f"추가근거3: {evidence_5}\n"
            f"테마: {summary}\n"
            "--------------------\n"
        )

    return context_data


def build_system_instruction(count_needed: int, exclude_names: list) -> str:
    exclude_text = ""
    if exclude_names:
        exclude_text = (
            "\n이미 추천된 장소는 다시 추천하지 마라.\n"
            "이미 추천된 장소:\n" + ", ".join(exclude_names) + "\n"
        )

    return f"""
너는 김해시 전문 여행 추천 AI다.

사용자의 여행 취향과 제공된 김해 관광지 데이터를 분석해서
가장 적합한 관광지 {count_needed}곳을 추천한다.

반드시 제공된 데이터에 존재하는 관광지만 추천해야 한다.
데이터에 없는 장소를 절대 만들지 마라.
데이터에 없는 시설, 비용, 운영시간, 주차정보,
무료 여부, 행사, 축제, 체험효과 등의 사실을
추측하거나 만들어내지 마라.

추천 이유는 반드시 제공된 관광지 데이터의
실제 근거와 의미적으로 연결되어야 한다.
{exclude_text}
출력은 반드시 JSON 배열 하나만 출력한다.

각 객체는 반드시 다음 5개 필드를 포함한다.
place_id
place_name
recommend_reason
matching_tags
image_prompt

place_id 규칙:
- 제공된 데이터의 place_id를 그대로 사용한다.
- 존재하지 않는 place_id를 만들지 않는다.

place_name 규칙:
- 제공된 데이터의 place_name을 그대로 사용한다.

recommend_reason 규칙:
- 반드시 {MIN_SENTENCES}문장 이상 작성한다.
- 반드시 {MAX_SENTENCES}문장 이하로 작성한다.
- 각 문장은 5자 이상의 완성된 문장이어야 한다.
- 각 문장의 끝에는 반드시 '.', '!', '?' 중 하나를 사용한다.
- 쉼표는 문장 구분자로 사용하지 않는다.
- 모든 내용은 제공된 데이터(객관적사실/주관적사실/추가근거/테마)에 근거해야 한다.
- 근거에 없는 내용을 추가하지 않는다.
- 가능하면 객관적사실·주관적사실·추가근거에 쓰인 표현과 어휘를 최대한 그대로
  활용해서 문장을 구성한다. 완전히 새로운 표현으로 막연하게 요약하지 말고,
  데이터에 있는 구체적인 표현을 우선한다.
- 테마(요약)에 쓰인 표현도 최소 한 문장 이상에 반영한다.
- 사용자가 피하고 싶다고 한 요소는 "어떤 장소를 고를지"에만 반영하고,
  문장 자체에는 "~와 달리", "~을 피하고 싶다면" 같은 부정형 비교 표현을
  쓰지 않는다. 각 문장은 선택된 장소에 대한 긍정적 서술로만 작성한다.

matching_tags 규칙:
- {MIN_TAGS}~{MAX_TAGS}개의 키워드를 리스트로 작성한다.
- 완성된 문장이 아니라 짧은 명사형 키워드로 작성한다.

image_prompt 규칙:
- 영문으로 작성한다.
- 해당 관광지에 관한 시각적 프롬프트를 작성한다.

동일한 장소를 중복 추천하지 않는다.
반드시 JSON 배열만 출력한다.
"""


def call_llm(user_pref: str, context_data: str, count_needed: int, exclude_names: list) -> list:
    system_instruction = build_system_instruction(count_needed, exclude_names)
    user_message = f"""
사용자의 여행 취향:

{user_pref}


추천 가능한 김해 관광지 데이터:

{context_data}


위 데이터를 분석해서 사용자의 취향에 가장 적합한
관광지 {count_needed}곳을 추천해라.

추천 이유는 반드시 위 데이터에 존재하는
실제 근거를 바탕으로, 가능한 한 그 표현을 살려서 작성해라.

데이터에 없는 사실은 절대로 추가하지 마라.
반드시 JSON 배열만 출력해라.
"""

    response = ollama.generate(
        model=OLLAMA_MODEL,
        system=system_instruction,
        prompt=user_message,
        options={"temperature": 0.1, "num_ctx": 8192},
    )

    return extract_json_array(response.get("response", ""))


# =========================================================
# Recommendation API
# =========================================================

def run_recommendation_pipeline(user_pref: str) -> list[dict]:
    if df_places is None or len(df_places) == 0:
        raise HTTPException(status_code=500, detail="관광지 데이터 파일을 불러올 수 없습니다.")

    if len(VALID_PLACES) == 0:
        raise HTTPException(status_code=500, detail="유효한 관광지 데이터가 없습니다.")

    pref = clean_text(user_pref)
    if not pref:
        raise HTTPException(status_code=400, detail="preference를 입력해주세요.")

    validated_list = []
    seen_ids = set()
    dev_log(f"추천 파이프라인 시작 - 취향 문장: {pref}")

    # ---- 1단계: LLM 최대 MAX_ATTEMPTS회 호출 ----
    for attempt in range(MAX_ATTEMPTS):
        needed = TARGET_COUNT - len(validated_list)
        if needed <= 0:
            break

        exclude_names = [VALID_PLACES[pid] for pid in seen_ids if pid in VALID_PLACES]
        context_data = build_context_data(df_places, exclude_ids=seen_ids)
        dev_log(f"1단계: Ollama({OLLAMA_MODEL}) 호출 시도 {attempt + 1}/{MAX_ATTEMPTS} - {needed}곳 추천 요청 (응답까지 수십 초 소요될 수 있음)")

        try:
            result_data = call_llm(
                user_pref=pref,
                context_data=context_data,
                count_needed=needed,
                exclude_names=exclude_names,
            )
        except Exception as e:
            print(f"[LLM 오류] 시도 {attempt + 1}/{MAX_ATTEMPTS}: {e}")
            dev_log(f"  LLM 호출 실패 (Ollama가 켜져 있는지 확인하세요): {e}")
            continue

        new_items = validate_and_fix_response(result_data, seen_ids, needed)
        dev_log(f"  LLM 응답 {len(result_data)}건 중 검증 통과 {len(new_items)}건: " + ", ".join(i["place_name"] for i in new_items))
        validated_list.extend(new_items)

        if len(validated_list) >= TARGET_COUNT:
            break

    # ---- 2단계: 그래도 부족하면 CSV 근거만으로 하드 백필 (최후 수단) ----
    if len(validated_list) < TARGET_COUNT:
        print(f"[보충] LLM 추천 결과 {len(validated_list)}/{TARGET_COUNT}")
        dev_log(f"2단계: LLM 결과가 {len(validated_list)}/{TARGET_COUNT}곳뿐이라 CSV 근거로 부족분을 보충합니다 (하드 백필)")

        remaining_ids = [pid for pid in VALID_PLACES if pid not in seen_ids]
        remaining_ids = rank_places_by_preference(pref, remaining_ids)

        for place_id in remaining_ids:
            backfill_item = create_hard_backfill_item(place_id)
            if not backfill_item:
                continue  # evidence가 부실한 장소는 억지로 넣지 않음

            validated_list.append(backfill_item)
            seen_ids.add(place_id)
            dev_log(f"  CSV 근거로 보충: {backfill_item['place_name']}")

            if len(validated_list) >= TARGET_COUNT:
                break

    # ---- 3단계: 최종 재검증 ----
    dev_log("3단계: 추천 결과 최종 재검증(장소ID/문장수/태그 확인) 중...")
    final_result = final_validate(validated_list)

    if len(final_result) < TARGET_COUNT:
        raise HTTPException(
            status_code=500,
            detail=(
                f"유효한 추천 결과를 {TARGET_COUNT}개 생성하지 못했습니다. "
                f"현재 추천 수: {len(final_result)}"
            ),
        )

    dev_log(f"3단계 최종 검증 완료: {len(final_result)}곳 확정 → " + ", ".join(i["place_name"] for i in final_result[:TARGET_COUNT]))
    return final_result[:TARGET_COUNT]


@app.post("/recommend")
async def recommend_places(req: UserRequest):
    return run_recommendation_pipeline(req.preference)


# =========================================================
# Preference & Region State (In-Memory & Cache)
# =========================================================

DEFAULT_PREFERENCE = "김해의 대표 명소"
latest_preference: str = ""
cached_recommendations: list[dict] = []
cached_preference: str = ""


def get_current_preference() -> str:
    """메모리에 저장된 최신 preference를 반환하며, 없으면 기본값을 반환한다."""
    return latest_preference if latest_preference else DEFAULT_PREFERENCE


def update_recommendation_cache(pref: str) -> list[dict]:
    """추천 파이프라인을 실행하고 결과를 캐시에 저장한다."""
    global cached_recommendations, cached_preference
    raw_recommendations = run_recommendation_pipeline(pref)
    cached_recommendations = [
        {
            "place_name": item["place_name"],
            "place_id": item["place_id"],
            "recommend_reason": item["recommend_reason"],
        }
        for item in raw_recommendations
    ]
    cached_preference = pref
    return cached_recommendations


@app.post("/preference")
async def set_preference(req: UserRequest):
    global latest_preference
    pref = clean_text(req.preference)
    if not pref:
        raise HTTPException(status_code=400, detail="preference를 입력해주세요.")
    latest_preference = pref

    # 취향 등록 시 즉시 캐시 갱신 (전시 연계 시 /current_region의 0초 즉각 응답 보장)
    try:
        update_recommendation_cache(latest_preference)
    except Exception as e:
        print(f"[WARN] 취향 변경 시 캐시 갱신 실패 (조회 시 재시도됨): {e}")

    return {
        "message": "성공적으로 저장되었습니다.",
        "preference": latest_preference,
    }


# =========================================================
# Current Region API
# =========================================================

@app.get("/current_region")
async def get_current_region(region_id: Optional[str] = None):
    # region_id 누락 또는 GIMHAE가 아닌 경우 (대소문자 무시 지원)
    clean_region = clean_text(region_id)
    if clean_region.upper() != "GIMHAE":
        return {
            "region_id": "GIMHAE",
            "message": "Requested region_id was not found."
        }

    # 캐시된 결과가 있으면 즉시 반환하여 전시 부스 HTTP 타임아웃 방지
    try:
        pref = get_current_preference()
        global cached_recommendations, cached_preference
        if cached_recommendations and cached_preference == pref:
            return {
                "region_id": "GIMHAE",
                "recommendations": cached_recommendations,
            }

        recommendations = update_recommendation_cache(pref)
        return {
            "region_id": "GIMHAE",
            "recommendations": recommendations,
        }
    except Exception as e:
        # LLM/파이프라인 오류 발생 시 기존 캐시가 있다면 폴백 반환
        if cached_recommendations:
            print(f"[WARN] 파이프라인 처리 중 오류가 발생하여 기존 캐시 결과를 반환합니다: {e}")
            return {
                "region_id": "GIMHAE",
                "recommendations": cached_recommendations,
            }

        error_msg = getattr(e, "detail", str(e))
        print(f"[ERROR] /current_region 오류 발생: {error_msg}")
        return {
            "region_id": "GIMHAE",
            "message": f"추천 파이프라인 처리 중 오류가 발생했습니다: {error_msg}"
        }


# =========================================================
# Mock: 영상 생성 모의 API + 프론트엔드
# =========================================================

from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from mock_video import router as mock_video_router, get_or_create_task
import uuid

# 2지선다 설문: 각 선택지는 취향 문장의 조각이 된다.
SURVEY_QUESTIONS = [
    {"question": "어떤 여행을 원하세요?",
     "options": ["고요하게 쉬는 힐링 여행", "몸을 움직이는 액티비티 여행"],
     "phrases": ["조용히 걸으며 힐링할 수 있는", "활동적으로 체험할 수 있는"]},
    {"question": "어떤 분위기가 끌리세요?",
     "options": ["역사와 전통이 있는 곳", "자연과 풍경이 좋은 곳"],
     "phrases": ["역사와 전통 문화유산이 있는", "자연 풍경이 아름다운"]},
    {"question": "누구와 함께 가나요?",
     "options": ["가족과 함께", "친구·연인과 함께"],
     "phrases": ["가족과 함께 가기 좋은", "친구나 연인과 함께 가기 좋은"]},
    {"question": "여행 중 무엇을 하고 싶으세요?",
     "options": ["여유롭게 산책", "새로운 체험과 배움"],
     "phrases": ["여유롭게 산책하기 좋은", "새로운 체험과 배움이 있는"]},
]


class SurveyRequest(BaseModel):
    answers: list[int]   # 문항별 선택지 인덱스(0/1)


@app.get("/api/survey/questions")
async def get_survey_questions():
    return [{"question": q["question"], "options": q["options"]} for q in SURVEY_QUESTIONS]


@app.post("/api/survey")
async def submit_survey(req: SurveyRequest):
    global latest_preference
    if len(req.answers) != len(SURVEY_QUESTIONS) or any(a not in (0, 1) for a in req.answers):
        raise HTTPException(status_code=400, detail="모든 문항에 답해주세요.")

    # 선택 키워드 -> 하나의 취향 문장
    phrases = [q["phrases"][a] for q, a in zip(SURVEY_QUESTIONS, req.answers)]
    preference = "김해에서 " + ", ".join(phrases) + " 관광지를 추천해주세요."

    dev_log(f"설문 응답 수신: 선택 {req.answers} → " + " / ".join(f"{q['question']} {q['options'][a]}" for q, a in zip(SURVEY_QUESTIONS, req.answers)))
    dev_log(f"선택을 취향 문장으로 조합: {preference}")
    latest_preference = preference   # /current_region(전시 서버 연계)에도 반영
    recommendations = await run_in_threadpool(update_recommendation_cache, preference)

    # 1순위 장소 영상 모의 작업을 미리 등록 (사용자가 카드를 읽는 동안 생성)
    session_id = uuid.uuid4().hex[:8]
    task_id, _ = get_or_create_task(recommendations[0]["place_id"], session_id)
    dev_log(f"설문 처리 완료 → 1순위 '{recommendations[0]['place_name']}' 영상 생성 작업({task_id}) 사전 등록, 화면에 추천 5곳 전달")
    return {
        "session_id": session_id,
        "preference": preference,
        "recommendations": recommendations,
        "task_id": task_id,
    }


@app.get("/api/dev-log")
async def get_dev_log(after: int = -1):
    """개발용 진행 로그 조회 (after 이후 id만 반환)"""
    return [e for e in logs if e["id"] > after]


app.include_router(mock_video_router)
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    # 명세서 필수 요구사항: 0.0.0.0:5000 바인드
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True, app_dir=BASE_DIR)