# 약품 정보 검색 및 질의응답 시스템

벡터 임베딩 검색과 Ollama Gemma3:4b LLM을 활용한 약품 정보 검색 및 질의응답 시스템입니다.

## 시스템 구성

### 주요 파일
- `main.py`: 메인 시스템 (벡터 검색 + LLM 통합)
- `retriever.py`: 벡터 임베딩 검색 모듈
- `llm_handler.py`: Ollama LLM 핸들러
- `load_data/config.py`: 데이터베이스 설정 및 임베딩 모델

### 데이터베이스
- PostgreSQL + pgvector를 사용한 벡터 데이터베이스
- 약품 정보 JSON 데이터를 벡터 임베딩으로 변환하여 저장

## 설치 및 설정

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. PostgreSQL + pgvector 설정
```bash
# Docker를 사용한 PostgreSQL + pgvector 실행
docker run --name postgres-vector -e POSTGRES_PASSWORD=admin123 -e POSTGRES_USER=admin -e POSTGRES_DB=vectordb -p 55432:5432 -d pgvector/pgvector:pg16
```

### 3. Ollama 설치 및 모델 다운로드
```bash
# Ollama 설치 (Windows)
# https://ollama.ai/download

# Gemma2:4b 모델 다운로드
ollama pull gemma2:4b
```

### 4. 데이터 로딩
```bash
# 약품 정보를 벡터 데이터베이스에 로딩
python load_data/main_loader.py
```

## 사용법

### 1. 기본 실행
```bash
python main.py
```

### 2. 테스트 모드
```bash
python main.py test
```

### 3. 개별 모듈 테스트
```bash
# Retriever 테스트
python retriever.py

# LLM 핸들러 테스트
python llm_handler.py
```

## 시스템 기능

### 1. 약품 검색
- 벡터 유사도 검색
- 키워드 검색
- 하이브리드 검색 (벡터 + 키워드)

### 2. 질의응답
- 컨텍스트 기반 답변 생성
- 약품 정보 전문가 역할

### 3. 약품 추천
- 증상 기반 약품 추천
- 사용법 및 주의사항 안내

### 4. 약물 상호작용 분석
- 여러 약품의 동시 복용 시 상호작용 분석
- 주의사항 및 권장사항 제시

### 5. 사용법 안내
- 특정 약품의 상세 사용법 안내
- 부작용 및 주의사항 설명

## 대화형 모드 명령어

시스템 실행 후 다음 명령어들을 사용할 수 있습니다:

```
검색: [검색어]          # 약품 검색
질문: [질문]            # 일반 질문
추천: [증상]            # 약품 추천
상호작용: [약품1, 약품2] # 약물 상호작용 분석
사용법: [약품명]        # 사용법 안내
quit 또는 exit          # 종료
```

## 예시 사용법

### 약품 검색
```
사용자: 검색: 두통에 좋은 약
시스템: [검색 결과 표시]
```

### 질문 답변
```
사용자: 질문: 게보린정은 어떤 약인가요?
시스템: [약품 정보 기반 답변]
```

### 약품 추천
```
사용자: 추천: 소화불량 증상
시스템: [증상에 맞는 약품 추천]
```

### 약물 상호작용 분석
```
사용자: 상호작용: 게보린정, 겔포스엠
시스템: [상호작용 분석 결과]
```

## 기술 스택

- **벡터 데이터베이스**: PostgreSQL + pgvector
- **임베딩 모델**: sentence-transformers (dragonkue/snowflake-arctic-embed-l-v2.0-ko)
- **LLM**: Ollama Gemma2:4b
- **프레임워크**: LangChain
- **언어**: Python 3.8+

## 주의사항

1. **의학적 조언**: 이 시스템은 참고용이며, 실제 의학적 조언은 의사나 약사와 상담하세요.
2. **모델 성능**: Gemma2:4b는 경량 모델이므로 복잡한 의학적 질문에는 한계가 있을 수 있습니다.
3. **데이터 정확성**: 약품 정보의 정확성을 위해 공식 의료 데이터베이스를 참조하세요.

## 문제 해결

### Ollama 연결 오류
- Ollama가 실행 중인지 확인: `ollama list`
- 모델이 설치되어 있는지 확인: `ollama pull gemma2:4b`

### 데이터베이스 연결 오류
- PostgreSQL이 실행 중인지 확인
- Docker 컨테이너 상태 확인: `docker ps`
- 연결 정보 확인 (load_data/config.py)

### 임베딩 모델 다운로드 오류
- 인터넷 연결 확인
- HuggingFace 토큰 설정 (필요시)
