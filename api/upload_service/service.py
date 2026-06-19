import logging
import os
from typing import Annotated, Any

import httpx
from fastapi import Depends
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings

from api.upload_service.constants import (
    PVSN_INST_GROUP_MAP,
    PLCY_PVSN_MTHD_MAP,
    PLCY_APRV_STTS_MAP,
    APLY_PRD_SE_MAP,
    BIZ_PRD_SE_MAP,
    MRG_STTS_MAP,
    EARN_CND_SE_MAP,
    PLCY_MAJOR_MAP,
    JOB_MAP,
    SCHOOL_MAP,
    SBIZ_MAP,
)
from api.upload_service.util import map_codes

# 온통청년 API KEY
YOUTH_API_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
YOUTH_API_KEY = os.getenv("YOUTH_API_KEY", "")

# 담당 분야 대분류 (본인 담당: 금융, 복지문화, 문화, 예술)
TARGET_LCLSF_NM = "복지문화"


class EmbeddingService:
    """
    온통청년 API 정책 데이터를 PGVector에 임베딩하여 저장하는 서비스.

    [수정 이력] 공식 명세서(oaiDoc) + 실제 호출 결과로 검증 완료:
      - 인증키 파라미터: openApiVlak -> apiKeyNm
      - 페이지 번호: pageIndex -> pageNum
      - 페이지 크기: display -> pageSize
      - 응답 구조: 최상위가 아니라 result.youthPolicyList 안에 있음 (실제 호출로 확인)
      - 페이지네이션: result.pagging.totCount로 전체 건수 확인 가능
      - 특화요건 필드: sBizCd (소문자 b, 실제 응답 기준)
      - 신청기간 필드: aplyYmd (응답값은 비어있는 경우가 많음 - 별도 확인 필요)
    """

    # 1. 초기화 메서드
    def __init__(self) -> None:
        self.logger = logging.getLogger(f"{__name__}.EmbeddingService")
        self.embeddings = init_embeddings(model="openai:text-embedding-3-large")

    # ------------------------------------------------------------------
    # 1) 온통청년 API 수집
    # ------------------------------------------------------------------

    async def fetch_youth_policy_page(
        self,
        lclsf_nm: str = "",
        page_num: int = 1,
        page_size: int = 100,
    ) -> dict[str, Any]:
        """온통청년 API에서 정책 목록 1페이지를 조회한다.

        Args:
            lclsf_nm: 정책대분류명 (콤마로 복수 지정 가능, 예: "복지문화,교육").
                빈 문자열이면 전체 분야를 대상으로 조회한다.
            page_num: 조회할 페이지 번호 (명세상 파라미터명: pageNum)
            page_size: 페이지당 결과 수 (명세상 파라미터명: pageSize)

        Returns:
            dict: API 응답 JSON
        """
        params: dict[str, Any] = {
            "apiKeyNm": YOUTH_API_KEY,
            "pageNum": page_num,
            "pageSize": page_size,
            "rtnType": "json",
        }
        if lclsf_nm:
            params["lclsfNm"] = lclsf_nm

        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(YOUTH_API_URL, params=params)
            response.raise_for_status()
            return response.json()

    async def fetch_all_youth_policies(
        self,
        lclsf_nm: str = TARGET_LCLSF_NM,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """페이지네이션으로 특정 대분류의 정책을 전부 수집한다.

        정책이 수백~천 건 단위라 한 번에 가져올 수 없으므로, 응답의
        result.pagging.totCount(전체 건수)를 기준으로 모든 페이지를 순회한다.
        totCount를 못 읽는 예외적인 경우를 대비해, 빈 페이지가 나오면도 중단한다.

        [수정] 응답이 최상위가 아니라 result 객체 안에 한 단계 더 감싸져 있음.
        실제 응답 구조:
            {
              "resultCode": 200,
              "result": {
                "pagging": {"totCount": 1046, "pageNum": 1, "pageSize": 5},
                "youthPolicyList": [ ... ]
              }
            }
        """
        page_num = 1
        all_items: list[dict[str, Any]] = []
        total_count: int | None = None

        while True:
            data = await self.fetch_youth_policy_page(lclsf_nm, page_num, page_size)
            result = data.get("result", {})
            items = result.get("youthPolicyList", [])

            if total_count is None:
                total_count = result.get("pagging", {}).get("totCount")
                if total_count is not None:
                    self.logger.info(f"{lclsf_nm or '전체'} - 전체 {total_count}건 확인됨")

            if not items:
                break

            all_items.extend(items)
            self.logger.info(f"{lclsf_nm or '전체'} - {page_num}페이지 수집 완료 ({len(items)}건)")

            # totCount를 알고 있고 이미 그만큼 모았으면 더 호출하지 않고 종료
            if total_count is not None and len(all_items) >= total_count:
                break

            page_num += 1

        self.logger.info(f"{lclsf_nm or '전체'} 전체 수집 완료: 총 {len(all_items)}건")
        return all_items

    def policy_to_document(self, policy: dict[str, Any]) -> Document:
        """정책 JSON 1건을 RAG 검색용 Document로 변환한다.

        metadata는 온통청년 API 원본 응답을 그대로 저장하되,
        코드값 필드는 사람이 읽을 수 있도록 코드명으로 변환하여 저장한다.
        """

        title = policy.get("plcyNm", "")
        explain_content = policy.get("plcyExplnCn", "")
        support_content = policy.get("plcySprtCn", "")
        apply_method = policy.get("plcyAplyMthdCn", "")

        text = (
            f"정책명: {title}\n"
            f"정책설명: {explain_content}\n"
            f"지원내용: {support_content}\n"
            f"신청방법: {apply_method}"
        )

        # 원본 응답 전체 저장
        metadata = dict(policy)

        # 코드 → 코드명 변환 (복수 코드 지원)
        metadata.update(
            {
                "category": policy.get("lclsfNm", ""),       # 정책대분류명
                "sub_category": policy.get("mclsfNm", ""),    # 정책중분류명
                "pvsnInstGroupCd": map_codes(
                    policy.get("pvsnInstGroupCd", ""),
                    PVSN_INST_GROUP_MAP,
                ),
                "plcyPvsnMthdCd": map_codes(
                    policy.get("plcyPvsnMthdCd", ""),
                    PLCY_PVSN_MTHD_MAP,
                ),
                "plcyAprvSttsCd": map_codes(
                    policy.get("plcyAprvSttsCd", ""),
                    PLCY_APRV_STTS_MAP,
                ),
                "aplyPrdSeCd": map_codes(
                    policy.get("aplyPrdSeCd", ""),
                    APLY_PRD_SE_MAP,
                ),
                "bizPrdSeCd": map_codes(
                    policy.get("bizPrdSeCd", ""),
                    BIZ_PRD_SE_MAP,
                ),
                "mrgSttsCd": map_codes(
                    policy.get("mrgSttsCd", ""),
                    MRG_STTS_MAP,
                ),
                "earnCndSeCd": map_codes(
                    policy.get("earnCndSeCd", ""),
                    EARN_CND_SE_MAP,
                ),
                "plcyMajorCd": map_codes(
                    policy.get("plcyMajorCd", ""),
                    PLCY_MAJOR_MAP,
                ),
                "jobCd": map_codes(
                    policy.get("jobCd", ""),
                    JOB_MAP,
                ),
                "schoolCd": map_codes(
                    policy.get("schoolCd", ""),
                    SCHOOL_MAP,
                ),
                # API 응답에 sBizCd, sbizCd 둘 다 존재하는 경우 대응
                "sBizCd": map_codes(
                    policy.get("sBizCd") or policy.get("sbizCd", ""),
                    SBIZ_MAP,
                )
            }
        )

        return Document(
            page_content=text,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # 2) 청크 분할
    # ------------------------------------------------------------------

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """Document를 청크 단위로 분할하고 중복을 제거한다."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
        )
        chunk_documents = text_splitter.split_documents(documents)

        seen: set[str] = set()
        unique_chunks: list[Document] = []
        for doc in chunk_documents:
            content = doc.page_content.strip()
            if content not in seen:
                seen.add(content)
                unique_chunks.append(doc)

        return unique_chunks

    # ------------------------------------------------------------------
    # 3) PGVector 저장 / 검색
    # ------------------------------------------------------------------

    async def save_to_vectorstore_with_sqlalchemy(
        self,
        collection_name: str,
        chunk_documents: list[Document],
    ) -> None:
        """청크 Document들을 임베딩하여 PGVector에 저장한다."""
        from api.common.sqlalchemy_conf import engine

        vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=engine,
            async_mode=True,
        )
        await vectorstore.aadd_documents(chunk_documents)
        self.logger.info(f"[{collection_name}] 임베딩 저장 완료 ({len(chunk_documents)}건)")


# 의존성 타입 별칭
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(EmbeddingService)]