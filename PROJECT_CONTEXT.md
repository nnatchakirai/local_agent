# 프로젝트 컨텍스트 & 개발 핸드오프 문서 (Hand-off Briefing)

```text
[다른 AI에게 전달할 때 사용할 복사 프롬프트]
아래는 우리 프로젝트의 전체 배경, API 명세, 현재 코드 구조, 그리고 앞으로 구현할 파이프라인(태블릿 설문 -> LoRA 영상 생성)이 정리된 컨텍스트 문서야. 
이 내용을 숙지하고 다음 작업을 진행해줘:
```

> **이 문서는 새로운 AI 모델이나 개발자가 프로젝트의 전체 배경, API 명세, 코드 구조, 그리고 차기 구현 파이프라인(태블릿 설문 ➡️ 추천 ➡️ LoRA 영상 ➡️ 상영)의 맥락을 즉시 파악할 수 있도록 정리된 공식 컨텍스트 문서입니다.**

---

## 1. 프로젝트 개요 & 배경
- **대회/행사명**: Re:Local 연계 본선 전시 및 실감콘텐츠 개발
- **목적**: 
  - 김해시 관광지 데이터(CSV) 기반의 개인화 여행지 추천 REST API 서버 구축
  - 현장 전시 연계 공통 규격(Re:Local REST API) 완벽 준수
  - 향후 관람객 태블릿 인터랙션(2지선다 설문)과 결합하여 생성형 AI(LoRA 이미지 및 비디오 생성)를 상영하는 실감형 전시 파이프라인 확장

---

## 2. Re:Local 공식 연계 REST API 규격 ([별첨] 명세 준수)

### (1) 기본 네트워크 규격
- **Protocol**: HTTP GET
- **Host / IP**: 팀별 할당 고정 IP (192.168.0.10x)
- **Port / Server Bind**: `0.0.0.0:5000` (**필수 요구사항**)
- **Encoding**: UTF-8 (JSON)
- **CORS**: 현장 웹 프론트엔드 연계를 위해 `Access-Control-Allow-Origin: *` 허용 필수

### (2) 핵심 엔드포인트 명세
- **Endpoint**: `GET /current_region?region_id={REGION_ID}`
- **파라미터**: `region_id` (Query Parameter, 대소문자 유연성 지원)
- **정상 응답 (HTTP 200 OK)**:
  ```json
  {
    "region_id": "GIMHAE",
    "recommendations": [
      {
        "place_name": "수로왕릉",
        "place_id": "GIMHAE_001",
        "recommend_reason": "수로왕릉은 가야의 건국 시조 수로왕의 능으로... (2~5문장)"
      },
      ... // 최소 5개 이상의 POI 필수
    ]
  }
  ```
  - **제약 조건**:
    - Root 필드: `region_id`, `recommendations`만 포함
    - 각 POI 필드: `place_name`, `place_id`, `recommend_reason` 3가지만 포함 (`matching_tags`, `image_prompt`는 외부 연계 응답 시 제외)
    - `place_id`는 제공된 CSV 데이터에 반드시 존재하는 ID 사용
    - `place_name`은 CSV 데이터의 명칭과 100% 일치
    - `recommend_reason`은 2~5문장 구성, 종결부호(`.`, `!`, `?`) 필수
- **오류 응답 권장 (HTTP 200 OK)**:
  ```json
  {
    "region_id": "GIMHAE",
    "message": "Requested region_id was not found."
  }
  ```

---

## 3. 백엔드 아키텍처 및 구현 상세 (`app.py`)

### (1) 핵심 파일 구성
- `app.py`: FastAPI 기반 추천 및 연계 API 서버
- `test_current_region.py`: 명세서 전 항목(프로토콜, 헤더, CORS, 오류케이스, 필드 검증, 문장 수, 캐시 응답속도) 자동 검증 테스트
- `trace_pipeline.py`: 원본 코드 수정 없이 LLM 추론부터 하드 백필까지 과정을 실시간 추적하는 스크립트
- `여행지_20_김해_수정본.csv`: 김해시 20개 관광지 기준 데이터셋 (객관/주관 근거 및 테마 요약 포함)

### (2) 현장 안정성(Fail-safe) 및 최적화 로직
1. **타임아웃 방지 메모리 캐싱**:
   - `POST /preference`로 취향이 등록되면 백그라운드/선제적으로 추천 결과를 계산하여 캐싱
   - 전시 시스템이 `GET /current_region` 호출 시 LLM 대기 시간 없이 **0.005초(5ms) 만에 즉각 응답**
2. **2단계 하드 백필 (Hard Backfill)**:
   - LLM 장애나 연결 오류 발생 시에도 서버가 다운되지 않고 CSV 근거 문장을 규칙에 맞게 조합하여 규격에 맞는 5개 추천 항목 자동 생성
3. **CORS 미들웨어 완비**:
   - `allow_origins=["*"]`, `allow_credentials=False` 설정으로 브라우저 CORS 차단 에러 사전 방지

---

## 4. 차기 확장 기획: 태블릿 인터랙션 & AI 미디어 생성 파이프라인

### (1) 서비스 시나리오
1. **태블릿 설문 (Frontend)**: 관람객이 2지선다 키워드 설문(3~4문항) 진행 (예: 고요한 힐링 vs 액티비티)
2. **취향 조합 (Backend)**: 선택된 키워드를 하나의 자연스러운 여행 취향 문장(`preference`)으로 조합
3. **개인화 추천 (LLM)**: OpenAI API를 통해 김해 명소 5곳 선정 및 `image_prompt` 생성
4. **결과 노출 & 백그라운드 생성 (UX 트릭)**:
   - 태블릿에는 5개 추천 명소 카드를 먼저 예쁜 애니메이션과 함께 띄워 관람객이 읽도록 유도
   - 관람객이 읽는 동안 백엔드는 **1순위 대표 명소**에 대해:
     - **Step 1**: LoRA 적용 이미지 생성 (SD / ComfyUI)
     - **Step 2**: Image-to-Video 변환 (SVD, Kling, Runway 등 이용해 3~5초 티저 비디오 렌더링)
5. **영상 상영**: 렌더링 완료 시 태블릿 화면이 전환되며 관람객 맞춤형 여행 영상 풀스크린 상영

### (2) 추가 구현할 API 엔드포인트 설계안
- `POST /api/survey`: 태블릿 설문 응답 수신 ➡️ 추천 결과 반환 & 비디오 생성 태스크 큐 등록 (`task_id` 발급)
- `GET /api/video-status/{task_id}`: 태블릿이 2초 간격으로 영상 렌더링 완료 여부 폴링
- `GET /current_region`: 방금 설문으로 갱신된 추천 데이터를 Re:Local 연계 메인 전시 서버가 실시간 호출하여 전시관 전체에 반영
