import logging
from typing import Annotated
from fastapi import APIRouter, Form, HTTPException, status
from fastapi.responses import PlainTextResponse
from api.upload_service.service import EmbeddingServiceDep


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/upload", tags=["upload"])


##################################################
# 온통청년 API 수집 + 임베딩 엔드포인트
##################################################
@router.post("/youth-policy", response_class=PlainTextResponse)
async def youth_policy_embedding(
    service: EmbeddingServiceDep,
    collection_name: Annotated[str, Form()] = "youth_policy_all",
    lclsf_nm: Annotated[str, Form()] = "복지문화",
):
    # 온통청년 API 전체 페이지네이션 수집
    # lclsf_nm은 정책대분류명 (콤마로 복수 지정 가능, 예: "복지문화,교육")
    policies = await service.fetch_all_youth_policies(lclsf_nm=lclsf_nm)

    if not policies:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{lclsf_nm}' 분야 정책 데이터를 찾을 수 없습니다.",
        )

    # JSON -> Document 변환
    documents = [service.policy_to_document(p) for p in policies]

    # 청크 단위로 분할
    chunks = service.split_documents(documents)

    # 벡터 저장소에 저장
    await service.save_to_vectorstore_with_sqlalchemy(
        collection_name=collection_name,
        chunk_documents=chunks,
    )

    # 결과 반환
    result = (
        f"✅ 청년정책 임베딩 완료!\n\n"
        f"- 컬렉션명: {collection_name}\n"
        f"- 대분류(lclsfNm): {lclsf_nm}\n"
        f"- 총 정책 수: {len(policies)}\n"
        f"- 총 청크 수: {len(chunks)}"
    )
    return result