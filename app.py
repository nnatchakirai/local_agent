import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama

app = FastAPI()

# 김해 관광지 데이터셋 로드
CSV_PATH = "여행지_20_김해_수정본.csv"
try:
    df_places = pd.read_csv(CSV_PATH)
    print("로컬 관광지 데이터 로드 성공!")
except Exception as e:
    df_places = None
    print(f" CSV 로드 실패: {e}. 'app.py'와 같은 폴더에 csv 파일이 있는지 확인하세요.")


# 사용자 요청 형식 정의
class UserRequest(BaseModel):
    preference: str  # 사용자의 취향 입력


# 추천 API 엔드포인트
@app.post("/recommend")
async def recommend_places(req: UserRequest):
    if df_places is None:
        raise HTTPException(status_code=500, detail="관광지 데이터 파일이 존재하지 않습니다.")

    user_pref = req.preference

    # 4. CSV 데이터셋을 AI에게 제공할 텍스트 형태로 결합
    context_data = ""
    for _, row in df_places.iterrows():
        context_data += (
            f"장소ID: {row['place_id']}\n"
            f"장소명: {row['place_name']}\n"
            f"객관적사실: {row['evidence_text_1']}\n"
            f"주관적사실: {row['evidence_text_2']}\n"
            f"테마: {row['summary']}\n"
            f"-------------------\n"
        )

    # 정량평가 감점 방지를 위한 시스템 지시문 (프롬프트 엔지니어링)
    system_instruction = (
        "너는 김해시 전문 여행 가이드 AI 에이전트야.\n"
        "제공된 관광지 데이터와 사용자의 취향을 분석해서 가장 적합한 5곳을 선정하고,\n"
        "반드시 지정된 JSON 형식으로만 답변해줘. 마크다운 기호(```json 등)나 부가 설명은 절대 하지마.\n\n"
        "각 문장은 반드시 제공된 객관적사실 또는 주관적사실 문장의 구체적인 표현이나 키워드를 그대로 활용해서 다시 서술해야 해. 새로운 표현으로 막연하게 꾸미지 마.\n"
        "[엄격한 정량평가 규칙]:\n"
        "1. 추천 장소의 'place_id'와 'place_name'은 반드시 제공된 데이터의 값을 토대로 100% 일치하게 적어야 해.\n"
        "2. 추천 사유('recommend_reason')는 반드시 최소 2문장 이상, 최대 5문장 이하여야 해.\n"
        "3. 추천 사유의 각 문장은 반드시 마침표(.), 느낌표(!), 물음표(?) 중 하나로 끝나야 해. (쉼표는 문장 구분 안 됨)\n"
        "4. 데이터에 없는 사실(예: 주차비, 구체적 요금, 지어낸 축제 정보 등)은 절대 쓰지 마.\n\n"
        "출력 예시:\n"
        "[\n"
        "  {\n"
        "    \"place_id\": \"GIMHAE_01\",\n"
        "    \"place_name\": \"김해 수로왕릉\",\n"
        "    \"recommend_reason\": \"수로왕릉은 가야의 역사를 느낄 수 있는 대표 유적지입니다. 한적하게 산책하며 역사 공부를 하기에 아주 좋습니다.\"\n"
        "  }\n"
        "]"
    )

    user_message = f"사용자 취향: {user_pref}\n\n김해 관광지 데이터:\n{context_data}"

    try:
        # 6. 로컬 Ollama에 탑재된 qwen2.5:7b 모델 호출
        response = ollama.generate(
            model="qwen2.5:7b",
            system=system_instruction,
            prompt=user_message,
            options={"temperature": 0.1, "num_ctx": 8192}  # 답변의 일관성을 높이고 환각을 줄이기 위해 낮춤
        )

        reply_text = response['response'].strip()
        print(f"[DEBUG] 원본 출력 길이: {len(reply_text)}자")
        print(f"[DEBUG] 원본 출력 내용: {reply_text!r}")

        # 코드블럭(```json ... ```)으로 감싸서 오는 경우 방어적으로 제거
        if reply_text.startswith("```"):
            reply_text = reply_text.strip("`")
            if reply_text.startswith("json"):
                reply_text = reply_text[4:].strip()

        
        # AI가 대답 앞뒤에 붙인 따옴표나 불필요한 공백 제거 후 JSON으로 변환
        result_data = json.loads(reply_text)
        return result_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"추론 중 오류 발생: {str(e)}")