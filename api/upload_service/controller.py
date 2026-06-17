import logging
from typing import Annotated
from fastapi import APIRouter, Form
from fastapi.responses import PlainTextResponse
from api.chat_service.langgraph.model import UserProfile
from api.upload_service.service import EmbeddingServiceDep

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/welfare", tags=["welfare"])


##################################################
# 복지 유사도 검색 (guest / user 테스트용)
##################################################
@router.post("/similarity-search", response_class=PlainTextResponse)
async def welfare_similarity_search(
    service: EmbeddingServiceDep,
    query: Annotated[str, Form()],
    k: Annotated[int, Form()] = 5,
    category: Annotated[str, Form()] = "복지",
    collection_name: Annotated[str, Form()] = "youth_policy_all",
    distance_threshold: Annotated[float, Form()] = 0.4,
    

    # 👉 추가: user 테스트용 프로필
    user_role: Annotated[str, Form()] = "guest",
    age: Annotated[int | None, Form()] = None,
    gender: Annotated[str | None, Form()] = None,
    is_university_graduate: Annotated[bool | None, Form()] = None,
    major: Annotated[str | None, Form()] = None,
    region: Annotated[str | None, Form()] = None,
    income_level: Annotated[int | None, Form()] = None,
):
    """
    guest / user 맞춤 검색 둘 다 테스트 가능한 엔드포인트
    """

    user_profile = None

    # user일 때만 profile 생성
    if user_role == "user":
        user_profile = UserProfile(
            age=age,
            gender=gender,
            is_university_graduate=is_university_graduate,
            major=major,
            region=region,
            income_level=income_level,
        )

    logger.info(f"[welfare-search] role={user_role}, profile={user_profile}")

    return await service.search_welfare_policy(
        query=query,
        category=category,
        collection_name=collection_name,
        k=k,
        distance_threshold=distance_threshold,
        user_role=user_role,
        user_profile=user_profile,
    )