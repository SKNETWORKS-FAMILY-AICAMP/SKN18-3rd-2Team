# 의약품 정보 Multi-Agent RAG 시스템 실행 순서

## 🚀 발표/데모용 실행 순서

### 1단계: 환경 준비
```bash
# Python 패키지 설치
pip install -r requirements.txt
```

### 2단계: Docker 서비스 시작
```bash
# PostgreSQL + Ollama 컨테이너 실행
docker-compose up -d

# 컨테이너 상태 확인 (선택사항)
docker-compose ps
```

### 3단계: Ollama 모델 설정 (자동)
```bash
# gemma3:4b 모델 자동 설치 및 설정
python setup_docker_ollama.py
```

### 4단계: 의약품 데이터 로딩
```bash
# CSV 데이터를 벡터 데이터베이스에 로딩
python load_csv_data.py
```

### 5단계: Multi-Agent RAG 시스템 실행
```bash
# 메인 시스템 실행 (데모용)
python test_multi_agent.py
```

---

## 📋 각 단계별 상세 설명

### 1단계: 환경 준비
- 필요한 Python 라이브러리들을 설치합니다
- 임베딩 모델, 데이터베이스 연결, HTTP 요청 등에 필요한 패키지들

### 2단계: Docker 서비스 시작
- **PostgreSQL + pgvector**: 벡터 데이터베이스
- **Ollama**: LLM 서버 (gemma3:4b 모델 실행용)
- 두 서비스가 모두 정상 실행되어야 함

### 3단계: Ollama 모델 설정
- Ollama 컨테이너에 gemma3:4b 모델 자동 설치
- 모델 다운로드 및 연결 테스트 수행
- 약 2-3분 소요 (모델 크기: ~2.5GB)

### 4단계: 의약품 데이터 로딩
- `data_set.csv` 파일의 의약품 정보를 읽어옴
- E5 모델로 임베딩 생성
- PostgreSQL 벡터 데이터베이스에 저장
- 약 5-10분 소요 (데이터 양에 따라)

### 5단계: 시스템 실행
- **대화형 모드**: 실시간 질문/답변
- **테스트 모드**: 미리 준비된 질문들로 시연
- Multi-Agent 분석 + LLM 종합 답변 생성

---

## ⚠️ 주의사항

1. **Docker 실행 확인**: `docker-compose ps`로 두 컨테이너 모두 "Up" 상태 확인
2. **모델 다운로드**: 3단계에서 인터넷 연결 필요 (약 2.5GB)
3. **데이터 로딩**: 4단계에서 시간이 걸릴 수 있음 (진행률 표시됨)
4. **메모리 사용량**: 임베딩 모델 + LLM 모델로 인한 높은 메모리 사용

---

## 🎯 발표 시 데모 시나리오

### 추천 질문들:
1. **증상 기반**: "두통에 좋은 약이 있나요?"
2. **용법/용량**: "게보린정은 하루에 몇 번 먹어야 하나요?"
3. **안전성**: "임신 중에 피해야 할 약물은 무엇인가요?"
4. **상호작용**: "타이레놀과 아스피린을 함께 먹어도 되나요?"

### 시연 포인트:
- Multi-Agent 분석 과정 보여주기
- 각 에이전트별 전문 분석 결과
- LLM의 종합 답변 생성
- 신뢰도 점수 및 참조 의약품 표시

---

## 🔧 문제 해결

### Docker 서비스 문제
```bash
# 서비스 재시작
docker-compose down
docker-compose up -d
```

### Ollama 연결 문제
```bash
# Ollama 컨테이너 로그 확인
docker logs drug_rag_ollama

# 모델 재설치
docker exec drug_rag_ollama ollama pull gemma3:4b
```

### 데이터베이스 문제
```bash
# PostgreSQL 컨테이너 로그 확인
docker logs drug_rag_postgres

# 데이터 재로딩
python load_csv_data.py
```