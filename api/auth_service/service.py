import logging
from typing import Annotated
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth_service.model import User
from api.common.sqlalchemy_conf import OrmSessionDep

logger = logging.getLogger(__name__)

# 유저 조회
class UserService:
    async def get_user_profile(self, user_id: int, session: AsyncSession) -> dict | None:
        result = await session.execute(select(User).where(User.user_id == user_id))
        user = result.scalar_one_or_none()
        if not user:
            return None
        logger.info("유저 프로필 조회 완료 — user_id=%s, nickname=%s", user.user_id, user.nickname)
        return {
            "user_id":      user.user_id,
            "nickname":     user.nickname,
            "birth_date":   str(user.birth_date),
            "zipcd":        user.zipcd,
            "category":     user.category,
            "schoolcd":     user.schoolcd,
            "plcymajorcd":  user.plcymajorcd,
            "jobcd":        user.jobcd,
            "mrgsttscd":    user.mrgsttscd,
            "sbizcd":       user.sbizcd,
            "earncndsecd":  user.earncndsecd,
        }


UserServiceDep = Annotated[UserService, Depends(UserService)]
