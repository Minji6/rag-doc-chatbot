import asyncio
import sys

# Windows 환경에서 psycopg async 호환 설정
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv
load_dotenv()

from api.chat_service.langgraph.agents.welfare_agent import WelfareAgent


async def main():
    agent = WelfareAgent()

    # ✅ UI 기반 mock user profile (실제 입력 구조 반영)
    user_profile = {
        "age": 24,
        "gender": "남성",
        
        "region": "서울",

        "income": {
            "min": 0,
            "max": 2000  # 저소득 기준
        },

        "education": "대학 재학",  # 고졸/대학재학/졸업
        "major_field": "공학계열",

        "employment_status": "미취업",

        "special_conditions": [
            "해당없음"
        ]
    }

    result = await agent.run(
        "저소득 청년을 위한 문화비 지원 정책 알려줘",
        user_role="user",
        user_profile=user_profile,   # dict로 전달
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())