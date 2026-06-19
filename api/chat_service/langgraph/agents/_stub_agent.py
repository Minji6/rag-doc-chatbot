import logging
from ..state import DomainResult


class StubAgent:
    """팀원 에이전트 완성 전까지 사용하는 임시 구현체."""
    def __init__(self, domain_label: str, category: str) -> None:
        self.logger = logging.getLogger(f"{__name__}.StubAgent.{domain_label}")
        self.domain_label = domain_label
        self.category = category

    async def run(self, question: str) -> DomainResult:
        self.logger.warning("%s 에이전트는 아직 stub 상태입니다.", self.domain_label)
        return DomainResult(
            text=f"[{self.domain_label} 에이전트 준비 중입니다. 곧 답변 가능합니다.]",
            policies=[],
            category=self.category,
            source="none",
        )
