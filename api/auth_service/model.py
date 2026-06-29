from datetime import date, datetime
from typing import Annotated
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Integer, String, Date, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from api.common.sqlalchemy_conf import Base


class UserCreateRequest(BaseModel):
    nickname:    Annotated[str, Field(max_length=50)]
    birth_date:  date
    zipcd:       Annotated[str | None, Field(None, max_length=50)]
    category:    Annotated[str | None, Field(None, max_length=100)]
    schoolcd:    Annotated[str | None, Field(None, max_length=50)]
    plcymajorcd: Annotated[str | None, Field(None, max_length=50)]
    jobcd:       Annotated[str | None, Field(None, max_length=50)]
    mrgsttscd:   Annotated[str | None, Field(None, max_length=50)]
    sbizcd:      Annotated[str | None, Field(None, max_length=200)]
    earncndsecd: int | None = None

    @field_validator("nickname")
    @classmethod
    def nickname_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("닉네임은 빈 값일 수 없습니다.")
        return v.strip()

    @field_validator("birth_date")
    @classmethod
    def birth_date_not_future(cls, v: date) -> date:
        if v < date(1900, 1, 1):
            raise ValueError("생년월일은 1900-01-01 이후여야 합니다.")
        if v > date.today():
            raise ValueError("생년월일은 오늘 이후 날짜로 등록할 수 없습니다.")
        return v


class UserResponse(BaseModel):
    user_id:     int
    nickname:    str
    birth_date:  date
    zipcd:       Annotated[str | None, Field(None)]
    category:    Annotated[str | None, Field(None)]
    schoolcd:    Annotated[str | None, Field(None)]
    plcymajorcd: Annotated[str | None, Field(None)]
    jobcd:       Annotated[str | None, Field(None)]
    mrgsttscd:   Annotated[str | None, Field(None)]
    sbizcd:      Annotated[str | None, Field(None)]
    earncndsecd: Annotated[int | None, Field(None)]
    model_config = {"from_attributes": True}


class User(Base):
    __tablename__ = "users"

    user_id:     Mapped[int]           = mapped_column(Integer, primary_key=True)
    nickname:    Mapped[str]           = mapped_column(String(50), nullable=False)
    birth_date:  Mapped[date]          = mapped_column(Date, nullable=False)
    zipcd:       Mapped[str | None]    = mapped_column(String(50))
    category:    Mapped[str | None]    = mapped_column(String(100))
    schoolcd:    Mapped[str | None]    = mapped_column(String(50))
    plcymajorcd: Mapped[str | None]    = mapped_column(String(50))
    jobcd:       Mapped[str | None]    = mapped_column(String(50))
    mrgsttscd:   Mapped[str | None]    = mapped_column(String(50))
    sbizcd:      Mapped[str | None]    = mapped_column(String(200))
    earncndsecd: Mapped[int | None]    = mapped_column(Integer)
    created_at:  Mapped[datetime]      = mapped_column(DateTime, server_default=func.now())
    updated_at:  Mapped[datetime]      = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
