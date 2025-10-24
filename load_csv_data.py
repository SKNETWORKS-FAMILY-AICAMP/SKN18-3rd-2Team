"""
CSV 데이터를 RAG 시스템에 로딩하는 스크립트
"""
import pandas as pd
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from src.rag_system import RAGSystem
from src.database import VectorDB

load_dotenv()

def load_csv_to_rag(csv_file: str = "data_set.csv", batch_size: int = 100):
    """CSV 파일을 RAG 시스템에 로딩"""
    
    # CSV 파일 존재 확인
    if not Path(csv_file).exists():
        print(f"❌ CSV 파일을 찾을 수 없습니다: {csv_file}")
        return False
    
    print(f"📄 CSV 파일 로딩: {csv_file}")
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"📊 총 {len(df)}개 행 발견")
        
        # 컬럼 확인
        print(f"📋 컬럼: {list(df.columns)}")
        
        # 필수 컬럼 확인 (유연하게 처리)
        content_col = None
        product_col = None
        
        # 가능한 컬럼명들 확인
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['content', '내용', 'description', '설명']):
                content_col = col
            elif any(keyword in col_lower for keyword in ['product', '제품', 'name', '이름', '품목']):
                product_col = col
        
        if not content_col or not product_col:
            print("❌ 필수 컬럼을 찾을 수 없습니다.")
            print("💡 CSV 파일에 다음 컬럼이 필요합니다:")
            print("   - 내용/설명 컬럼 (content, 내용, description 등)")
            print("   - 제품명 컬럼 (product, 제품, name, 품목 등)")
            return False
        
        print(f"✅ 내용 컬럼: {content_col}")
        print(f"✅ 제품명 컬럼: {product_col}")
        
        # RAG 시스템 초기화
        print("\n🚀 RAG 시스템 초기화 중...")
        rag_system = RAGSystem()
        
        # 기존 데이터 확인
        db = VectorDB()
        cursor = db.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents;")
        existing_count = cursor.fetchone()[0]
        
        if existing_count > 0:
            print(f"⚠️ 기존 데이터 {existing_count}개 발견")
            overwrite = input("기존 데이터를 삭제하고 새로 로딩하시겠습니까? (y/N): ").lower()
            if overwrite == 'y':
                cursor.execute("DELETE FROM documents;")
                db.connection.commit()
                print("🗑️ 기존 데이터 삭제 완료")
            else:
                print("📝 기존 데이터에 추가로 로딩합니다.")
        
        db.close()
        
        # 배치 단위로 데이터 로딩
        print(f"\n📥 데이터 로딩 시작 (배치 크기: {batch_size})")
        
        successful_loads = 0
        failed_loads = 0
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            print(f"📦 배치 {i//batch_size + 1}/{(len(df)-1)//batch_size + 1} 처리 중...")
            
            for idx, row in batch.iterrows():
                try:
                    content = str(row[content_col]).strip()
                    product_name = str(row[product_col]).strip()
                    
                    # 빈 데이터 스킵
                    if not content or content == 'nan' or not product_name or product_name == 'nan':
                        continue
                    
                    # 너무 짧은 내용 스킵
                    if len(content) < 10:
                        continue
                    
                    # RAG 시스템에 추가
                    doc_id = rag_system.add_document(content, product_name)
                    successful_loads += 1
                    
                    if successful_loads % 50 == 0:
                        print(f"   ✅ {successful_loads}개 문서 로딩 완료")
                    
                except Exception as e:
                    failed_loads += 1
                    if failed_loads <= 5:  # 처음 5개 오류만 출력
                        print(f"   ❌ 행 {idx} 로딩 실패: {e}")
        
        print(f"\n🎉 데이터 로딩 완료!")
        print(f"✅ 성공: {successful_loads}개")
        print(f"❌ 실패: {failed_loads}개")
        
        # 로딩 결과 확인
        products = rag_system.get_products()
        print(f"📋 등록된 제품 수: {len(products)}")
        
        if len(products) > 0:
            print(f"📝 제품 예시: {products[:5]}")
        
        rag_system.close()
        return True
        
    except Exception as e:
        print(f"❌ CSV 로딩 실패: {e}")
        return False

def test_loaded_data():
    """로딩된 데이터 테스트"""
    print("\n🧪 로딩된 데이터 테스트")
    print("-" * 30)
    
    try:
        rag_system = RAGSystem()
        
        # 테스트 쿼리들
        test_queries = [
            "두통에 좋은 약이 있나요?",
            "게보린정에 대해 알려주세요",
            "아스피린의 부작용은 무엇인가요?"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}] {query}")
            answer = rag_system.query(query)
            print(f"답변: {answer[:200]}...")
        
        rag_system.close()
        print("\n✅ 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")

def show_data_statistics():
    """데이터 통계 표시"""
    try:
        db = VectorDB()
        cursor = db.connection.cursor()
        
        # 총 문서 수
        cursor.execute("SELECT COUNT(*) FROM documents;")
        total_docs = cursor.fetchone()[0]
        
        # 제품 수
        cursor.execute("SELECT COUNT(DISTINCT product_name) FROM documents;")
        total_products = cursor.fetchone()[0]
        
        # 평균 문서 길이
        cursor.execute("SELECT AVG(LENGTH(content)) FROM documents;")
        avg_length = cursor.fetchone()[0]
        
        print(f"📊 데이터베이스 통계:")
        print(f"   📄 총 문서 수: {total_docs:,}")
        print(f"   🏷️ 제품 수: {total_products:,}")
        print(f"   📏 평균 문서 길이: {avg_length:.0f}자")
        
        # 상위 제품들
        cursor.execute("""
            SELECT product_name, COUNT(*) as doc_count 
            FROM documents 
            GROUP BY product_name 
            ORDER BY doc_count DESC 
            LIMIT 5
        """)
        
        top_products = cursor.fetchall()
        print(f"\n🔝 문서가 많은 제품 TOP 5:")
        for product, count in top_products:
            print(f"   {product}: {count}개 문서")
        
        db.close()
        
    except Exception as e:
        print(f"❌ 통계 조회 실패: {e}")

if __name__ == "__main__":
    print("🚀 CSV 데이터 로딩 시스템")
    print("=" * 50)
    
    # 현재 데이터 상태 확인
    show_data_statistics()
    
    # CSV 파일 확인
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("❌ CSV 파일을 찾을 수 없습니다.")
        print("💡 data_set.csv 파일을 프로젝트 루트에 배치해주세요.")
        exit(1)
    
    print(f"\n📄 발견된 CSV 파일: {csv_files}")
    
    # 기본 파일 또는 선택
    if len(csv_files) == 1:
        selected_file = csv_files[0]
    else:
        print("\n사용할 CSV 파일을 선택하세요:")
        for i, file in enumerate(csv_files, 1):
            print(f"{i}. {file}")
        
        choice = input(f"선택 (1-{len(csv_files)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            selected_file = csv_files[int(choice) - 1]
        else:
            selected_file = csv_files[0]
    
    print(f"✅ 선택된 파일: {selected_file}")
    
    # 데이터 로딩
    print("\n" + "=" * 50)
    success = load_csv_to_rag(selected_file)
    
    if success:
        # 로딩 후 통계
        print("\n" + "=" * 50)
        show_data_statistics()
        
        # 테스트 실행 여부
        test_choice = input("\n로딩된 데이터를 테스트하시겠습니까? (Y/n): ").lower()
        if test_choice != 'n':
            test_loaded_data()
        
        print("\n🎉 데이터 로딩 완료!")
        print("\n다음 단계:")
        print("1. python test_multi_agent.py (Multi-Agent 시스템 테스트)")
        print("2. python example_usage.py (기본 RAG 시스템 테스트)")
    else:
        print("\n❌ 데이터 로딩 실패")
        print("💡 CSV 파일 형식을 확인하거나 데이터베이스 연결을 확인해주세요.")