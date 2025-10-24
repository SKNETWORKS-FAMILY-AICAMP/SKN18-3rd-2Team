#!/usr/bin/env python3
"""
의약품 정보 Multi-Agent RAG 시스템 - 올인원 실행 파일
이 파일 하나만 실행하면 모든 설정과 시스템이 자동으로 실행됩니다.
"""
import subprocess
import sys
import time
import requests
import os
from pathlib import Path

class DrugRAGSystem:
    """의약품 RAG 시스템 올인원 실행기"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.steps_completed = []
        
    def print_header(self, title):
        """헤더 출력"""
        print(f"\n{'='*60}")
        print(f"🚀 {title}")
        print(f"{'='*60}")
    
    def print_step(self, step_num, title):
        """단계 출력"""
        print(f"\n📋 [{step_num}/7] {title}")
        print("-" * 40)
    
    def run_command(self, command, description, timeout=300, show_output=False):
        """명령어 실행"""
        print(f"🔄 {description}...")
        
        try:
            if show_output:
                # 실시간 출력
                process = subprocess.Popen(
                    command, 
                    shell=True, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=self.project_root
                )
                
                for line in process.stdout:
                    print(f"   {line.rstrip()}")
                
                process.wait()
                success = process.returncode == 0
            else:
                # 백그라운드 실행
                result = subprocess.run(
                    command, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=timeout,
                    cwd=self.project_root
                )
                success = result.returncode == 0
                
                if not success:
                    print(f"   오류: {result.stderr}")
            
            if success:
                print(f"✅ {description} 완료")
                return True
            else:
                print(f"❌ {description} 실패")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ {description} 타임아웃 ({timeout}초)")
            return False
        except Exception as e:
            print(f"❌ {description} 오류: {e}")
            return False
    
    def check_prerequisites(self):
        """사전 요구사항 확인"""
        self.print_step(1, "사전 요구사항 확인")
        
        # Python 버전 확인
        python_version = sys.version_info
        if python_version.major < 3 or python_version.minor < 8:
            print(f"❌ Python 3.8+ 필요 (현재: {python_version.major}.{python_version.minor})")
            return False
        print(f"✅ Python 버전: {python_version.major}.{python_version.minor}")
        
        # Docker 확인
        try:
            result = subprocess.run(['docker', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"✅ Docker 설치됨")
            else:
                print("❌ Docker가 설치되지 않았습니다.")
                return False
        except:
            print("❌ Docker가 설치되지 않았습니다.")
            print("💡 https://docs.docker.com/get-docker/ 에서 Docker를 설치하세요.")
            return False
        
        # 필수 파일 확인
        required_files = [
            'requirements.txt', 'docker-compose.yml', 'data_set.csv',
            'setup_docker_ollama.py', 'load_csv_data.py', 'test_multi_agent.py'
        ]
        
        for file in required_files:
            if not (self.project_root / file).exists():
                print(f"❌ 필수 파일 누락: {file}")
                return False
        print("✅ 필수 파일 확인 완료")
        
        self.steps_completed.append("prerequisites")
        return True
    
    def install_packages(self):
        """Python 패키지 설치"""
        self.print_step(2, "Python 패키지 설치")
        
        success = self.run_command(
            "pip install -r requirements.txt", 
            "Python 패키지 설치", 
            timeout=180
        )
        
        if success:
            self.steps_completed.append("packages")
        return success
    
    def start_docker_services(self):
        """Docker 서비스 시작"""
        self.print_step(3, "Docker 서비스 시작")
        
        # 기존 컨테이너 정리
        print("🧹 기존 컨테이너 정리 중...")
        subprocess.run(['docker-compose', 'down'], 
                      capture_output=True, cwd=self.project_root)
        
        # 서비스 시작
        success = self.run_command(
            "docker-compose up -d", 
            "PostgreSQL + Ollama 컨테이너 시작", 
            timeout=120
        )
        
        if success:
            self.steps_completed.append("docker")
        return success
    
    def wait_for_services(self):
        """서비스 준비 대기"""
        self.print_step(4, "서비스 준비 대기")
        
        print("⏳ PostgreSQL 준비 대기 중...")
        for i in range(30):
            try:
                # PostgreSQL 연결 테스트
                sys.path.append(str(self.project_root))
                from src.database import VectorDB
                db = VectorDB()
                db.close()
                print("✅ PostgreSQL 준비 완료")
                break
            except:
                if i % 5 == 0:
                    print(f"   대기 중... ({i}/30초)")
                time.sleep(1)
        else:
            print("❌ PostgreSQL 준비 실패")
            return False
        
        print("⏳ Ollama 준비 대기 중...")
        for i in range(60):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    print("✅ Ollama 준비 완료")
                    self.steps_completed.append("services")
                    return True
            except:
                pass
            
            if i % 10 == 0:
                print(f"   대기 중... ({i}/60초)")
            time.sleep(1)
        
        print("❌ Ollama 준비 실패")
        return False
    
    def setup_ollama_model(self):
        """Ollama 모델 설정"""
        self.print_step(5, "Ollama 모델 설정")
        
        print("📥 gemma3:4b 모델 설치 중... (시간이 걸릴 수 있습니다)")
        success = self.run_command(
            "python setup_docker_ollama.py", 
            "Ollama 모델 설정", 
            timeout=600,
            show_output=True
        )
        
        if success:
            self.steps_completed.append("ollama")
        return success
    
    def load_data(self):
        """의약품 데이터 로딩"""
        self.print_step(6, "의약품 데이터 로딩")
        
        print("📊 의약품 데이터 로딩 중... (시간이 걸릴 수 있습니다)")
        success = self.run_command(
            "python load_csv_data.py", 
            "데이터 로딩", 
            timeout=600,
            show_output=True
        )
        
        if success:
            self.steps_completed.append("data")
        return success
    
    def run_system(self):
        """Multi-Agent RAG 시스템 실행"""
        self.print_step(7, "Multi-Agent RAG 시스템 실행")
        
        print("🎉 모든 설정이 완료되었습니다!")
        print("🤖 Multi-Agent RAG 시스템을 시작합니다...")
        print("\n" + "="*60)
        
        try:
            # 시스템 실행
            os.chdir(self.project_root)
            subprocess.run([sys.executable, "test_multi_agent.py"])
            self.steps_completed.append("system")
            return True
        except KeyboardInterrupt:
            print("\n\n👋 시스템을 종료합니다.")
            return True
        except Exception as e:
            print(f"❌ 시스템 실행 오류: {e}")
            return False
    
    def cleanup_on_failure(self):
        """실패 시 정리 작업"""
        print("\n🧹 정리 작업 중...")
        
        if "docker" in self.steps_completed:
            print("🛑 Docker 서비스 중지 중...")
            subprocess.run(['docker-compose', 'down'], 
                          capture_output=True, cwd=self.project_root)
    
    def show_manual_steps(self):
        """수동 실행 방법 안내"""
        print("\n💡 수동 실행 방법:")
        print("1. pip install -r requirements.txt")
        print("2. docker-compose up -d")
        print("3. python setup_docker_ollama.py")
        print("4. python load_csv_data.py")
        print("5. python test_multi_agent.py")
        print("\n📋 자세한 내용은 RUN_ORDER.md 파일을 참조하세요.")
    
    def run_all(self):
        """전체 실행 프로세스"""
        self.print_header("의약품 정보 Multi-Agent RAG 시스템 올인원 실행")
        
        print("🚀 이 스크립트는 다음 작업을 자동으로 수행합니다:")
        print("   1. 사전 요구사항 확인")
        print("   2. Python 패키지 설치")
        print("   3. Docker 서비스 시작")
        print("   4. 서비스 준비 대기")
        print("   5. Ollama 모델 설정")
        print("   6. 의약품 데이터 로딩")
        print("   7. Multi-Agent RAG 시스템 실행")
        
        print(f"\n⏱️ 예상 소요 시간: 10-15분 (인터넷 속도에 따라)")
        
        # 사용자 확인
        try:
            response = input("\n계속 진행하시겠습니까? (Y/n): ").lower()
            if response == 'n':
                print("👋 실행을 취소합니다.")
                return False
        except KeyboardInterrupt:
            print("\n👋 실행을 취소합니다.")
            return False
        
        # 단계별 실행
        steps = [
            self.check_prerequisites,
            self.install_packages,
            self.start_docker_services,
            self.wait_for_services,
            self.setup_ollama_model,
            self.load_data,
            self.run_system
        ]
        
        try:
            for step in steps:
                if not step():
                    print(f"\n❌ 실행 실패: {step.__name__}")
                    self.cleanup_on_failure()
                    self.show_manual_steps()
                    return False
            
            print("\n🎉 모든 과정이 성공적으로 완료되었습니다!")
            return True
            
        except KeyboardInterrupt:
            print("\n\n👋 실행을 중단합니다.")
            self.cleanup_on_failure()
            return False
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류: {e}")
            self.cleanup_on_failure()
            self.show_manual_steps()
            return False

def main():
    """메인 함수"""
    system = DrugRAGSystem()
    success = system.run_all()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()