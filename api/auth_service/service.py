import logging
from typing import Annotated
from fastapi import Depends
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth_service.model import User, UserCreateRequest, UserResponse
from api.common.sqlalchemy_conf import OrmSessionDep

logger = logging.getLogger(__name__)


class UserService:
    async def get_users(self, session: AsyncSession) -> list[dict]:
        result = await session.execute(select(User).order_by(User.user_id))
        users = result.scalars().all()
        logger.info("유저 목록 조회 완료 — %d명", len(users))
        return [UserResponse.model_validate(u).model_dump() for u in users]

    async def get_user_profile(self, user_id: int, session: AsyncSession) -> dict | None:
        result = await session.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return None
        logger.info("유저 프로필 조회 완료 — user_id=%s, nickname=%s", user.user_id, user.nickname)
        return UserResponse.model_validate(user).model_dump()

    async def create_user(self, session: AsyncSession, req: UserCreateRequest) -> dict:
        user = User(**req.model_dump())
        session.add(user)
        await session.flush()
        await session.refresh(user)
        logger.info("유저 생성 완료 — user_id=%s, nickname=%s", user.user_id, user.nickname)
        return {"user_id": user.user_id, "nickname": user.nickname}


    async def get_conversations(self, user_id: str, session: AsyncSession) -> list[dict]:
        rows = await session.execute(
            text("""
                SELECT thread_id, MAX(checkpoint_id) AS latest_checkpoint
                FROM checkpoints
                WHERE thread_id LIKE :prefix
                GROUP BY thread_id
                ORDER BY latest_checkpoint DESC
            """),
            {"prefix": f"{user_id}:%"},
        )
        records = rows.fetchall()
        prefix = f"{user_id}:"
        logger.info("유저 %s 대화 목록 조회 — %d건", user_id, len(records))
        return [{"conversation_id": row[0].removeprefix(prefix)} for row in records]

    async def delete_user(self, user_id: int, session: AsyncSession) -> dict | None:
        user = (await session.execute(select(User).where(User.user_id == user_id))).scalar_one_or_none()
        if not user:
            return None

        # LangGraph 연관 테이블 삭제 (thread_id 형식: "{user_id}:{conversation_id}")
        prefix = f"{user_id}:%"
        for table in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
            await session.execute(
                text(f"DELETE FROM {table} WHERE thread_id LIKE :prefix"),
                {"prefix": prefix},
            )

        await session.delete(user)
        logger.info("유저 삭제 완료 — user_id=%s", user_id)
        return {"user_id": user_id, "message": "유저가 삭제되었습니다."}


UserServiceDep = Annotated[UserService, Depends(UserService)]
