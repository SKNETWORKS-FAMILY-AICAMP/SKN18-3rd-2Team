#!/usr/bin/env python3
"""
의약품 정보 Multi-Agent RAG 시스템 데모 실행 스크립트
"""
import subprocess
import sys
import time
import requests

def run_command(command, description, timeout=300):
    """명령어 실행"""
    print(f"\n🔄 {description}...")
    print(f"   명령어: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode == 0:
            print(f"✅ {description} 완료")
            return True
        else:
            print(f"❌ {description} 실패")
            print(f"   오류: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ {description} 타임아웃 ({timeout}초)")
        return False
    except Exception as e:
        print(f"❌ {description} 오류: {e}")
        return False

def check_docker():
    """Docker 상태 확인"""
    print("🐳 Docker 상태 확인...")
    
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Docker 설치됨")
            return True
        else:
            print("❌ Docker가 설치되지 않았습니다.")
            return False
    except:
        print("❌ Docker가 설치되지 않았습니다.")
        return False

def wait_for_services():
    """서비스 준비 대기"""
    print("\n⏳ 서비스 준비 대기 중...")
    
    # PostgreSQL 대기
    for i in range(30):
        try:
            from src.database import VectorDB
            db = VectorDB()
            db.close()
            print("✅ PostgreSQL 준비 완료")
            break
        except:
            if i % 5 == 0:
                print(f"   PostgreSQL 대기 중... ({i}/30초)")
            time.sleep(1)
    else:
        print("❌ PostgreSQL 준비 실패")
        return False
    
    # Ollama 대기
    for i in range(60):
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama 준비 완료")
                return True
        except:
            pass
        
        if i % 10 == 0:
            print(f"   Ollama 대기 중... ({i}/60초)")
        time.sleep(1)
    
    print("❌ Ollama 준비 실패")
    return False

def main():
    """메인 실행 함수"""
    print("🚀 의약품 정보 Multi-Agent RAG 시스템 데모")
    print("=" * 60)
    
    # 1. Docker 확인
    if not check_docker():
        print("\n💡 Docker를 먼저 설치해주세요.")
        return False
    
    # 2. 패키지 설치
    if not run_command("pip install -r requirements.txt", "Python 패키지 설치", 120):
        print("\n💡 패키지 설치에 실패했습니다. 가상환경을 확인해주세요.")
        return False
    
    # 3. Docker 서비스 시작
    if not run_command("docker-compose up -d", "Docker 서비스 시작", 120):
        print("\n💡 Docker 서비스 시작에 실패했습니다.")
        return False
    
    # 4. 서비스 준비 대기
    if not wait_for_services():
        print("\n💡 서비스 준비에 실패했습니다.")
        return False
    
    # 5. Ollama 모델 설정
    print("\n📥 Ollama 모델 설정 중... (시간이 걸릴 수 있습니다)")
    if not run_command("python setup_docker_ollama.py", "Ollama 모델 설정", 600):
        print("\n💡 Ollama 설정에 실패했습니다.")
        return False
    
    # 6. 데이터 로딩
    print("\n📊 의약품 데이터 로딩 중... (시간이 걸릴 수 있습니다)")
    if not run_command("python load_csv_data.py", "데이터 로딩", 600):
        print("\n💡 데이터 로딩에 실패했습니다.")
        return False
    
    # 7. 시스템 실행
    print("\n🎉 설정 완료! Multi-Agent RAG 시스템을 시작합니다...")
    print("\n" + "=" * 60)
    
    try:
        subprocess.run("python test_multi_agent.py", shell=True)
    except KeyboardInterrupt:
        print("\n\n👋 시스템을 종료합니다.")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n❌ 데모 실행에 실패했습니다.")
            print("💡 RUN_ORDER.md 파일을 참조하여 수동으로 실행해주세요.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 데모 실행을 중단합니다.")
        sys.exit(0)