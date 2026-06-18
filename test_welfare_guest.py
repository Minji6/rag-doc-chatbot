import asyncio
import sys

# Windows에서 psycopg 비동기 모드를 쓰려면 이벤트 루프 정책을 바꿔야 함
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from dotenv import load_dotenv

load_dotenv()

from api.chat_service.langgraph.agents.welfare_agent import WelfareAgent


async def main():
    agent = WelfareAgent()
    result = await agent.run("저소득 청년을 위한 문화비 지원 정책 알려줘", user_role="guest")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())