# 약품 정보 데이터 로더

## 📁 구조 (2개 파일)
```
load_data/
├── config.py        # DB 설정, JSON 로더, 임베딩 모델
└── main_loader.py   # 메인 로딩 로직
```

## 🚀 사용법
```bash
cd load_data
python main_loader.py
```

## 📊 처리 과정
1. JSON → JSONLoader로 문서 로드
2. RecursiveJsonSplitter로 청킹 (300자 기준)
3. 임베딩 생성 → VectorDB 저장
4. 제품명 메타데이터 포함

## 📋 데이터 구조
- **소스**: `drug_info_preprocessed.json`
- **메타데이터**: 제품명
- **벡터 차원**: 384차원
- **DB**: VectorDB만 사용 (RDB 없음)

## 🔧 주요 변경사항
- ❌ CSV 기반 → ✅ JSON 기반
- ❌ RDB + VectorDB → ✅ VectorDB만
- ❌ 면접 QA → ✅ 약품 정보
- ✅ RecursiveJsonSplitter 사용
- ✅ 제품명 메타데이터 추가



# 기존 데이터 삭제하고 새로 삽입
cd load_data
python main_loader.py --clear  # clear_mode=True로 실행