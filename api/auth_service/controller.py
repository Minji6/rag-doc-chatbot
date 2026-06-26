import logging
from fastapi import APIRouter, HTTPException, status

from api.auth_service.model import UserCreateRequest
from api.auth_service.service import UserServiceDep
from api.common.sqlalchemy_conf import OrmSessionDep

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.get("/users")
async def get_users(service: UserServiceDep, session: OrmSessionDep):
    return await service.get_users(session)


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(body: UserCreateRequest, service: UserServiceDep, session: OrmSessionDep):
    return await service.create_user(session, body)


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, service: UserServiceDep, session: OrmSessionDep):
    result = await service.delete_user(user_id, session)
    if result is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"user_id={user_id} 유저를 찾을 수 없습니다.")
    return result
