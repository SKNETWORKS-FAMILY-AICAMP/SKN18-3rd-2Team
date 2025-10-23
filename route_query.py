#워크 플로우 1의, 사용자의 입력값이 약에 대한 정보인지 아닌지 판정하는 라우터 함수(노드)
import ollama

# 사용할 LLM 모델 (gemma 계열)
OLLAMA_MODEL = 'gemma3:4b'

def route_query(user_query: str) -> str:
    """
    사용자의 질문을 분석하여 가장 적절한 전문가 에이전트의 유형을 결정합니다.
    """
    
    # LLM에게 역할을 부여하고 선택지를 제공하는 프롬프트
    prompt = f"""
    당신은 사용자 질문의 의도를 파악하고, 그에 맞는 전문가를 지정하는 라우터(Router) AI입니다.
    사용자의 질문을 읽고, 아래 네 가지 전문가 유형 중 가장 적절한 것 하나만 선택하여 그 이름만 정확히 답변해주세요.
    다른 어떤 설명도 붙이지 마세요.

    [전문가 유형]
    1. symptom_agent: 사용자가 아픈 증상을 말하며 약을 추천해달라고 요청할 때. (예: "머리가 아픈데 무슨 약 먹어야 해?")
    2. drug_info_agent: 사용자가 특정 약의 이름이나 정보를 물어볼 때. (예: "타이레놀 효능 알려줘.")
    3. side_effect_agent: 사용자가 여러 약과 증상을 함께 말하며 부작용을 의심할 때. (예: "감기약이랑 소화제 먹었는데 속이 쓰려.")
    4. irrelevant: 위 세 가지 경우에 해당하지 않는 모든 질문. (예: "오늘 날씨 어때?")

    [사용자 질문]
    "{user_query}"

    [답변]
    가장 적절한 전문가 유형:
    """

    print(f"1. 라우팅 시작: '{user_query}'")

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        category = response['message']['content'].strip()
        print(f"   - 라우팅 결과: {category}")
        return category
    
    except Exception as e:
        print(f"❌ 라우팅 중 오류 발생: {e}")
        # 오류 발생 시 안전하게 'irrelevant'로 처리
        return "irrelevant"

# --- 테스트를 위한 실행 코드 ---
if __name__ == "__main__":
    # 테스트 케이스
    query1 = "PGVector를 구현하려는데 어떻게 해야 할 지 잘 모르겠어. 설명해줘."
    query2 = "불타는 아이스 아메리카노를 마셨는데 머리가 커졌어"
    query3 = "마귀광대버섯이 거울에 비치고 오늘은 좀 가벼운 것 같은데 어떻게 생각해?"
    query4 = "바흐의 녹턴 교향곡 제 13장이 화성인에게 주는 증상이 뭔지 알아?"

    print("\n--- 테스트 1 ---")
    route_query(query1) # 예상 결과: symptom_agent

    print("\n--- 테스트 2 ---")
    route_query(query2) # 예상 결과: drug_info_agent

    print("\n--- 테스트 3 ---")
    route_query(query3) # 예상 결과: side_effect_agent

    print("\n--- 테스트 4 ---")
    route_query(query4) # 예상 결과: irrelevant