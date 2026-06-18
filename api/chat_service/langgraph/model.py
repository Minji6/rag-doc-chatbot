from pydantic import BaseModel, Field
from typing import Optional, Literal

class BasicInfo(BaseModel):
    age: Optional[int] = Field(None, description="나이")
    gender: Optional[Literal["남성", "여성", "기타"]] = Field(None, description="성별")
    region: Optional[str] = Field(None, description="거주 지역")

class EconomicInfo(BaseModel):
    income_level: Optional[int] = Field(
        None,
        description="기준 중위소득 비율(%)"
    )

    employment_status: Optional[
        Literal[
            "무직",
            "미취업",
            "재직자",
            "자영업자",
            "프리랜서",
            "단기근로",
            "창업자"
        ]
    ] = Field(None, description="취업 상태")

class EducationInfo(BaseModel):
    education_level: Optional[
        Literal[
            "고졸 미만",
            "고졸",
            "대학 재학",
            "대학 졸업",
            "대학원 재학",
            "대학원 졸업",
            "기타"
        ]
    ] = Field(None, description="학력")

    major: Optional[str] = Field(None, description="전공")

class SpecialCondition(BaseModel):
    is_low_income: Optional[bool] = Field(None, description="저소득 여부")
    is_disabled: Optional[bool] = Field(None, description="장애 여부")
    is_veteran: Optional[bool] = Field(None, description="보훈 대상 여부")
    is_single_parent: Optional[bool] = Field(None, description="한부모 여부")
    is_farmer: Optional[bool] = Field(None, description="농업 종사 여부")

    company_type: Optional[
        Literal["중소기업", "대기업", "공공기관", "스타트업", "해당없음"]
    ] = Field("해당없음", description="근무 기업 유형")


class UserProfile(BaseModel):
    basic: BasicInfo = BasicInfo() #type: ignore
    economic: EconomicInfo = EconomicInfo() #type: ignore
    education: EducationInfo = EducationInfo() #type: ignore
    special: SpecialCondition = SpecialCondition() #type: ignore