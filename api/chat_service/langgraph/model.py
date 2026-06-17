from typing import Annotated

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """사용자 맞춤 검색에 활용되는 프로필 정보.

    현재는 실제 사용자 스펙이 확정되지 않은 상태이므로, 테스트 및 기능 검증을 위해
    나이, 성별, 대학 졸업 여부, 전공의 4개 필드만 우선 정의한다.

    guest 사용자는 이 모델 없이(None) 동작할 수 있으며, 추후 서비스 요구사항이
    구체화되면 지역, 소득, 재직 상태 등 추가 정보가 Optional 필드로 확장될 수 있다.
    추가 필드는 자격 판별 에이전트 및 관련 도메인과 협의하여 점진적으로 반영한다.
    """

    age: Annotated[int | None, Field(description="나이", default=None)]
    gender: Annotated[str | None, Field(description="성별 (남성/여성/기타)", default=None)]
    is_university_graduate: Annotated[
        bool | None, Field(description="대학 졸업 여부", default=None)
    ]
    major: Annotated[str | None, Field(description="전공", default=None)]

    # 추가될 수 있는 필드 (현재는 모두 Optional, 확정되는 대로 채움)
    region: Annotated[str | None, Field(description="거주 지역", default=None)]
    income_level: Annotated[int | None, Field(description="기준 중위소득 비율(%)", default=None)]