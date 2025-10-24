"""
Docker Ollama 설정 자동화 스크립트
"""
import subprocess
import requests
import time
import json

def check_docker():
    """Docker 설치 및 실행 상태 확인"""
    print("🐳 Docker 상태 확인 중...")
    
    try:
        # Docker 설치 확인
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✅ Docker 설치됨: {result.stdout.strip()}")
        else:
            print("❌ Docker가 설치되지 않았습니다.")
            return False
        
        # Docker 실행 상태 확인
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ Docker 서비스 실행 중")
            return True
        else:
            print("❌ Docker 서비스가 실행되지 않았습니다.")
            print("💡 Docker Desktop을 시작하거나 'sudo systemctl start docker' 실행")
            return False
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Docker가 설치되지 않았습니다.")
        print("💡 https://docs.docker.com/get-docker/ 에서 Docker를 설치하세요.")
        return False

def start_docker_services():
    """Docker Compose로 서비스 시작"""
    print("🚀 Docker 서비스 시작 중...")
    
    try:
        # 기존 컨테이너 정리
        print("🧹 기존 컨테이너 정리 중...")
        subprocess.run(['docker-compose', 'down'], 
                      capture_output=True, timeout=30)
        
        # 서비스 시작
        print("⏳ PostgreSQL + Ollama 컨테이너 시작 중...")
        result = subprocess.run(['docker-compose', 'up', '-d'], 
                              capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Docker 서비스 시작 완료!")
            return True
        else:
            print(f"❌ Docker 서비스 시작 실패: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Docker 서비스 시작 타임아웃")
        return False
    except Exception as e:
        print(f"❌ Docker 서비스 시작 오류: {e}")
        return False

def wait_for_ollama():
    """Ollama 컨테이너 준비 대기"""
    print("⏳ Ollama 컨테이너 준비 대기 중...")
    
    for i in range(60):  # 최대 60초 대기
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama 컨테이너 준비 완료!")
                return True
        except:
            pass
        
        if i % 10 == 0:
            print(f"   대기 중... ({i}/60초)")
        time.sleep(1)
    
    print("❌ Ollama 컨테이너 준비 타임아웃")
    return False

def install_gemma3_model():
    """Docker Ollama에 gemma3:4b 모델 설치"""
    print("📥 gemma3:4b 모델 설치 중...")
    
    try:
        # 이미 설치된 모델 확인
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if 'gemma3:4b' in model_names:
                print("✅ gemma3:4b 모델이 이미 설치되어 있습니다.")
                return True
        
        # Docker 컨테이너에서 모델 설치
        print("⏳ 모델 다운로드 중... (시간이 걸릴 수 있습니다)")
        result = subprocess.run([
            'docker', 'exec', 'drug_rag_ollama', 
            'ollama', 'pull', 'gemma3:4b'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            print("✅ gemma3:4b 모델 설치 완료!")
            return True
        else:
            print(f"❌ 모델 설치 실패: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ 모델 설치 타임아웃 (10분)")
        return False
    except Exception as e:
        print(f"❌ 모델 설치 오류: {e}")
        return False

def test_docker_ollama():
    """Docker Ollama 테스트"""
    print("🧪 Docker Ollama 테스트 중...")
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma3:4b",
                "prompt": "안녕하세요. 간단히 인사해주세요.",
                "stream": False,
                "options": {"max_tokens": 50}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result.get('response', '')
            print(f"🤖 테스트 응답: {llm_response[:100]}...")
            print("✅ Docker Ollama 테스트 성공!")
            return True
        else:
            print(f"❌ 테스트 실패: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        return False

def show_container_status():
    """컨테이너 상태 표시"""
    print("\n📊 컨테이너 상태:")
    try:
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("컨테이너 상태를 확인할 수 없습니다.")
    except Exception as e:
        print(f"상태 확인 오류: {e}")

def main():
    """메인 설정 함수"""
    print("🚀 Docker Ollama 설정 자동화")
    print("=" * 50)
    
    # 1. Docker 확인
    if not check_docker():
        return False
    
    # 2. Docker 서비스 시작
    if not start_docker_services():
        return False
    
    # 3. Ollama 준비 대기
    if not wait_for_ollama():
        return False
    
    # 4. 모델 설치
    if not install_gemma3_model():
        return False
    
    # 5. 테스트
    if not test_docker_ollama():
        return False
    
    # 6. 상태 표시
    show_container_status()
    
    print("\n🎉 Docker Ollama 설정 완료!")
    print("\n📋 유용한 명령어:")
    print("   docker-compose up -d     # 서비스 시작")
    print("   docker-compose down      # 서비스 중지")
    print("   docker-compose logs      # 로그 확인")
    print("   docker-compose ps        # 상태 확인")
    
    print("\n다음 단계:")
    print("   1. python test_ollama_connection.py (연결 테스트)")
    print("   2. python test_multi_agent.py (Multi-Agent 시스템 실행)")
    
    return True

if __name__ == "__main__":
    if main():
        # 추가 테스트 실행 여부
        test_choice = input("\n연결 테스트를 실행하시겠습니까? (Y/n): ").lower()
        if test_choice != 'n':
            print("\n" + "=" * 50)
            import os
            os.system("python test_ollama_connection.py")