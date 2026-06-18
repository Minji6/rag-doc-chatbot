import logging
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, UploadFile, status
from fastapi.responses import PlainTextResponse

from langchain_core.document_loaders import Blob
from langchain_community.document_loaders.parsers.pdf import PyPDFParser

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


##################################################
# PDF 보강 임베딩 엔드포인트 (정책 데이터가 부족할 때 사용)
##################################################
@router.post("/pdf-embedding", response_class=PlainTextResponse)
async def pdf_embedding(
    title: Annotated[str, Form()],
    author: Annotated[str, Form()],
    attach: Annotated[UploadFile, Form()],
    service: EmbeddingServiceDep,
    region: Annotated[str, Form()] = "",
    collection_name: Annotated[str, Form()] = "youth_policy_all",
):
    # 파일 검증
    if attach.content_type != "application/pdf":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="PDF 파일만 업로드 가능")

    # PDF 로드 및 임베딩 처리
    content = await attach.read()
    blob = Blob.from_data(content, mime_type="application/pdf")
    parser = PyPDFParser()
    documents = list(parser.lazy_parse(blob))

    # 메타데이터 추가
    # PDF는 CSV처럼 분류 컬럼이 없으므로 category는 자동 추론이 안 됨.
    # 정책 데이터가 부족해 PDF로 보강하는 경우 source로 구분만 해두고,
    # category가 필요하면 업로드 시 별도 파라미터로 받아 추가하는 걸 고려.
    documents = service.add_metadata(
        documents,
        title=title,
        author=author,
        region=region,
        source=attach.filename,  # type: ignore
    )

    # 청크 단위로 분할
    chunks = service.split_documents(documents)

    # 벡터 저장소에 저장
    await service.save_to_vectorstore_with_sqlalchemy(
        collection_name=collection_name,
        chunk_documents=chunks,
    )

    # 결과 반환
    result = (
        f"✅ PDF 임베딩 완료!\n\n"
        f"- 컬렉션명: {collection_name}\n"
        f"- 제목: {title}\n"
        f"- 작성자: {author}\n"
        f"- 총 페이지 수: {len(documents)}\n"
        f"- 총 청크 수: {len(chunks)}"
    )
    return result


##################################################
# CSV 보강 임베딩 엔드포인트 (정책 데이터가 부족할 때 사용)
##################################################
@router.post("/csv-embedding", response_class=PlainTextResponse)
async def csv_embedding(
    title: Annotated[str, Form()],
    region: Annotated[str, Form()],
    attach: Annotated[UploadFile, Form()],
    service: EmbeddingServiceDep,
    collection_name: Annotated[str, Form()] = "youth_policy_all",
):
    # 파일 검증
    if not attach.filename or not attach.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV 파일만 업로드 가능")

    content = await attach.read()

    documents = service.load_csv_from_bytes(
        content=content,
        region=region,
    )

    documents = service.add_metadata(
        documents,
        title=title,
        source=attach.filename,
    )

    chunks = service.split_documents(documents)

    await service.save_to_vectorstore_with_sqlalchemy(
        collection_name=collection_name,
        chunk_documents=chunks,
    )

    # 결과 반환
    result = (
        f"✅ CSV 임베딩 완료!\n\n"
        f"- 컬렉션명: {collection_name}\n"
        f"- 지역: {region}\n"
        f"- 총 row 수: {len(documents)}\n"
        f"- 총 청크 수: {len(chunks)}"
    )
    return result