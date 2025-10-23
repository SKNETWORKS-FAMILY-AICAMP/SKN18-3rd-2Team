"""
RAG 기반 약품 정보 시스템
벡터 검색 + Self-RAG 에이전트 통합
"""
import os
import sys
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# 로컬 모듈 임포트
from RAG.langgraph import run_medicine_rag
from RAG.model import llm


def main():
    """메인 함수 - RAG 기반 약품 정보 시스템 테스트"""
    print("=== RAG 기반 약품 정보 시스템 ===")
    
    try:
        # 1. 모델 로드 확인
        print("모델 로드 확인 중...")
        print(f"LLM 모델: {type(llm).__name__}")
        print("임베딩 모델: HuggingFace")
        print("모델 로드 완료")
        
        # 2. 테스트 질문들
        test_questions = [
            "게보린정에 대해 알려주세요",
            "알파간피 점안액에 대해 알려주세요",
            "감기약 추천해주세요",
            "사랑해"
        ]
        
        print("\n=== 테스트 시작 ===")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- 테스트 {i} ---")
            print(f"질문: {question}")
            
            try:
                # 단순화된 RAG 실행
                result = run_medicine_rag(question)
                
                print(f"답변: {result.get('final_answer', '답변을 생성할 수 없습니다.')}")
                print("질문 처리 완료")
                
            except Exception as e:
                print(f"질문 처리 실패: {e}")
        
        print("\n=== 테스트 완료 ===")
        
    except Exception as e:
        print(f"시스템 초기화 실패: {e}")
        print("다음을 확인해주세요:")
        print("1. PostgreSQL 데이터베이스가 실행 중인지")
        print("2. 필요한 Python 패키지들이 설치되어 있는지")
        print("3. 데이터가 로드되어 있는지")
    
    finally:
        print("시스템 종료")


if __name__ == "__main__":
    main()