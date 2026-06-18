import asyncio
import logging
import os
import selectors
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import Depends
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import init_embeddings


# 온통청년 API 관련 상수
# TODO: .env로 옮기는 걸 권장 (YOUTH_API_KEY=...로 저장 후 os.getenv로 읽기)
#
# [확정] 공식 명세서(oaiDoc) 기준 - URL은 원래 코드가 맞았음
YOUTH_API_URL = "https://www.youthcenter.go.kr/go/ythip/getPlcy"
YOUTH_API_KEY = os.getenv("YOUTH_API_KEY", "")

# 담당 분야 대분류 (본인 담당: 금융, 복지문화, 문화, 예술)
# [확정] lclsfNm은 요청 파라미터로 그대로 사용 가능. 콤마로 여러 대분류 동시 요청 가능.
# (API코드정보.xlsx "정책대분류" 시트 기준 값: 일자리/주거/교육/복지문화/참여권리)
TARGET_LCLSF_NM = "복지문화"


class EmbeddingService:
    """
    정책/문서 데이터를 PGVector에 임베딩하여 저장하는 서비스.

    참고 코드(EnbeddingService, PDF 단일 파일 임베딩)를 기반으로
    다음 두 가지 데이터 소스를 모두 처리할 수 있도록 확장함:
      1. 온통청년 API (정책 JSON) - 1차 데이터 소스
      2. PDF/CSV 파일 - 정책 데이터가 부족할 때의 보강 소스

    [수정 이력] 공식 명세서(oaiDoc) + 실제 호출 결과로 검증 완료:
      - 인증키 파라미터: openApiVlak -> apiKeyNm
      - 페이지 번호: pageIndex -> pageNum
      - 페이지 크기: display -> pageSize
      - 응답 구조: 최상위가 아니라 result.youthPolicyList 안에 있음 (실제 호출로 확인)
      - 페이지네이션: result.pagging.totCount로 전체 건수 확인 가능
      - 특화요건 필드: sbizCd (소문자 b, 실제 응답 기준)
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

        텍스트는 "정책명: ... 지원내용: ... 신청방법: ..." 형태로 구성하고,
        metadata에는 상세조회/자격판별 에이전트가 활용할 필드를 담는다.

        필드명은 공식 명세서(oaiDoc) 기준으로 확정됨.
        """
        title = policy.get("plcyNm", "")
        # [확정] 정책설명(plcyExplnCn)과 정책지원내용(plcySprtCn)이 별도 필드로 존재.
        # 설명은 plcyExplnCn, 구체적인 지원 내용은 plcySprtCn에 들어있으므로 둘 다 합쳐서 사용.
        explain_content = policy.get("plcyExplnCn", "")
        support_content = policy.get("plcySprtCn", "")
        apply_method = policy.get("plcyAplyMthdCn", "")

        text = (
            f"정책명: {title}\n"
            f"정책설명: {explain_content}\n"
            f"지원내용: {support_content}\n"
            f"신청방법: {apply_method}"
        )

        metadata = {
            "policy_id": policy.get("plcyNo", ""),
            "category": policy.get("lclsfNm", ""),       # 정책대분류명
            "sub_category": policy.get("mclsfNm", ""),    # 정책중분류명
            "title": title,
            "earn_condition_code": policy.get("earnCndSeCd", ""),  # 소득조건 구분
            "special_biz_code": policy.get("sbizCd", ""),          # 정책특화요건코드
            # 신청기간: aplyYmd가 비어있는 경우가 많아(실제 호출 결과로 확인됨)
            # 신청기간구분코드(특정기간/상시/마감)와 사업기간도 함께 보관
            "apply_period": policy.get("aplyYmd", ""),
            "apply_period_type_code": policy.get("aplyPrdSeCd", ""),  # 0057001:특정기간/0057002:상시/0057003:마감
            "biz_period_start": policy.get("bizPrdBgngYmd", "").strip(),
            "biz_period_end": policy.get("bizPrdEndYmd", "").strip(),
            "apply_url": policy.get("aplyUrlAddr", ""),             # 신청 URL (출처 명시에 활용)
            "region_code": policy.get("zipCd", ""),                 # 정책거주지역코드 (지역 필터링 대비)
            "min_age": policy.get("sprtTrgtMinAge", ""),             # 지원대상최소연령 (자격판별 활용)
            "max_age": policy.get("sprtTrgtMaxAge", ""),             # 지원대상최대연령 (자격판별 활용)
        }

        return Document(page_content=text, metadata=metadata)

    # ------------------------------------------------------------------
    # 2) PDF / CSV 보강 (참고 코드 기반, 메서드명/시그니처 유지)
    # ------------------------------------------------------------------

    async def load_pdf(self, file_path: str) -> list[Document]:
        """PDF 파일을 로드해 Document 리스트를 반환한다."""
        if not Path(file_path).exists():
            raise FileNotFoundError("파일을 찾을 수 없습니다.")
        loader = PyPDFLoader(file_path)
        documents = await loader.aload()
        return documents

    def _detect_encoding(self, file_path: str) -> str:
        """CSV 파일의 인코딩을 자동 감지한다.

        한국에서 만든 CSV는 UTF-8이 아니라 CP949(EUC-KR)나
        UTF-8 BOM으로 저장된 경우가 많아, "utf-8"로 고정해 읽으면
        UnicodeDecodeError가 발생할 수 있다. charset-normalizer로
        실제 인코딩을 감지하고, 실패 시 cp949로 폴백한다.
        """
        from charset_normalizer import from_path

        result = from_path(file_path).best()
        if result is not None and result.encoding:
            return result.encoding
        return "cp949"

    # 분류 컬럼명은 지자체 CSV마다 다를 수 있음 (확인된 값: 대전 "정책분야", 울산 "정책성격")
    CATEGORY_FIELD_CANDIDATES = ["정책분야", "정책성격", "지원분야", "분류"]

    # 4개 표준 카테고리(교육/주거/일자리/복지)로 매핑.
    # 확인된 고유값(대전·울산 동일): 교육/주거/창업지원/취업지원/복지·문화/참여·권리
    CATEGORY_MAPPING: dict[str, str] = {
        "교육": "교육",
        "주거": "주거",
        "창업지원": "일자리",
        "취업지원": "일자리",
        "복지·문화": "복지",
        "참여·권리": "복지",
    }

    def _extract_field(self, page_content: str, field_names: list[str]) -> str:
        """CSVLoader가 만든 'key: value' 형태의 page_content에서 필드 값을 추출한다."""
        lines = page_content.split("\n")
        for field_name in field_names:
            for line in lines:
                if line.startswith(f"{field_name}:"):
                    return line.split(":", 1)[1].strip()
        return ""

    def _normalize_category(self, raw_category: str) -> str:
        """원본 CSV의 분류 표현을 4개 표준 카테고리로 통일한다."""
        for keyword, standard_category in self.CATEGORY_MAPPING.items():
            if keyword in raw_category:
                return standard_category
        if raw_category:
            self.logger.warning(f"매핑되지 않은 분류값 발견: '{raw_category}' -> '기타'로 분류됨")
        return "기타"

    def load_csv(self, file_path: str, region: str = "") -> list[Document]:
        """CSV 파일을 로드해 Document 리스트를 반환한다."""
        if not Path(file_path).exists():
            raise FileNotFoundError("파일을 찾을 수 없습니다.")

        detected_encoding = self._detect_encoding(file_path)
        self.logger.info(f"감지된 인코딩: {detected_encoding}")

        try:
            loader = CSVLoader(file_path=file_path, encoding=detected_encoding)
            documents = loader.load()
        except UnicodeDecodeError:
            self.logger.warning(f"{detected_encoding} 디코딩 실패, cp949로 재시도")
            loader = CSVLoader(file_path=file_path, encoding="cp949")
            documents = loader.load()

        for doc in documents:
            raw_category = self._extract_field(doc.page_content, self.CATEGORY_FIELD_CANDIDATES)
            doc.metadata["category"] = self._normalize_category(raw_category)
            doc.metadata["region"] = region
            doc.metadata["source"] = "public_data_csv"

        return documents

    def add_metadata(self, documents: list[Document], **metadata: str) -> list[Document]:
        """Document 리스트에 공통 메타데이터를 추가한다."""
        if metadata:
            for doc in documents:
                for key, value in metadata.items():
                    doc.metadata[key] = value
        return documents

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

    async def similarity_search(
        self,
        collection_name: str,
        query: str,
        k: int = 3,
    ) -> str:
        """저장된 벡터에서 유사도 검색을 수행한다(연동 테스트용)."""
        from api.common.sqlalchemy_conf import engine

        vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name=collection_name,
            connection=engine,
            async_mode=True,
        )
        results = await vectorstore.asimilarity_search_with_score(query, k)

        if not results:
            return "관련된 문서를 찾을 수 없습니다."
        return "\n".join(
            f"거리: {distance}, 내용: {doc.page_content[:30]}" for doc, distance in results
        )

    # ------------------------------------------------------------------
    # 4) 전체 파이프라인 (lifespan에서 호출할 진입점)
    # ------------------------------------------------------------------

    async def run_youth_policy_pipeline(
        self,
        collection_name: str = "youth_policy_all",
        lclsf_nm: str = TARGET_LCLSF_NM,
    ) -> None:
        """온통청년 API 전체 수집 -> Document 변환 -> 청크 분할 -> PGVector 저장.

        FastAPI lifespan 이벤트에서 앱 시작 시 1회 호출하는 것을 전제로 작성.
        """
        policies = await self.fetch_all_youth_policies(lclsf_nm=lclsf_nm)

        if not policies:
            self.logger.warning(f"{lclsf_nm} 분야 정책 데이터가 없습니다. 보강 소스 검토 필요.")
            return

        documents = [self.policy_to_document(p) for p in policies]
        chunk_documents = self.split_documents(documents)

        await self.save_to_vectorstore_with_sqlalchemy(
            collection_name=collection_name,
            chunk_documents=chunk_documents,
        )

    async def run_file_pipeline(
        self,
        file_path: str,
        collection_name: str,
        region: str = "",
        **extra_metadata: str,
    ) -> None:
        """PDF/CSV 파일을 임베딩하는 보강 파이프라인 (정책 데이터가 부족할 때 사용)."""
        suffix = Path(file_path).suffix.lower()

        if suffix == ".pdf":
            documents = await self.load_pdf(file_path)
        elif suffix == ".csv":
            documents = self.load_csv(file_path, region=region)
        else:
            raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")

        if extra_metadata:
            documents = self.add_metadata(documents, **extra_metadata)

        chunk_documents = self.split_documents(documents)
        await self.save_to_vectorstore_with_sqlalchemy(
            collection_name=collection_name,
            chunk_documents=chunk_documents,
        )


# 의존성 타입 별칭
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(EmbeddingService)]


# 최상위 모듈로 직접 실행하는 경우 (연동 테스트용)
if __name__ == "__main__":
    import sys

    sys.path.append(str(Path(__file__).parents[2]))

    logging.basicConfig(level=logging.INFO)
    service = EmbeddingService()

    def _selector_loop() -> asyncio.AbstractEventLoop:
        return asyncio.SelectorEventLoop(selectors.SelectSelector())

    # 1) 온통청년 API 파이프라인 테스트 (본인 담당 분야: 복지문화)
    asyncio.run(
        service.run_youth_policy_pipeline(
            collection_name="youth_policy_all",
            lclsf_nm="복지문화",
        ),
        loop_factory=_selector_loop,
    )

    # 2) 검색 테스트
    result = asyncio.run(
        service.similarity_search(
            collection_name="youth_policy_all",
            query="저소득 청년을 위한 금융 지원 정책이 있나요?",
            k=3,
        ),
        loop_factory=_selector_loop,
    )
    print(result)