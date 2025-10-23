"""
시퀀스 리셋 스크립트
"""
import psycopg2
from pgvector.psycopg2 import register_vector

# DB 설정
DB_CONFIG = {
    'host': 'localhost',
    'port': 55432,
    'database': 'vectordb', 
    'user': 'admin',
    'password': 'admin123'
}

def reset_sequence():
    """시퀀스를 1부터 다시 시작하도록 리셋"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    cursor = conn.cursor()
    
    try:
        # 현재 최대 ID 확인
        cursor.execute("SELECT MAX(id) FROM qa_embedding")
        max_id = cursor.fetchone()[0]
        print(f"현재 최대 ID: {max_id}")
        
        # 시퀀스를 최대 ID + 1로 설정
        cursor.execute(f"SELECT setval('qa_embedding_id_seq', {max_id})")
        conn.commit()
        
        print("시퀀스 리셋 완료")
        print("다음 삽입 시 ID는 1부터 시작됩니다.")
        
    except Exception as e:
        print(f"오류: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    reset_sequence()
