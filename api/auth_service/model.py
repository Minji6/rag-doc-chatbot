from datetime import date
from sqlalchemy import Integer, String, Date
from sqlalchemy.orm import Mapped, mapped_column
from api.common.sqlalchemy_conf import Base


class User(Base):
    __tablename__ = "users"

    user_id:               Mapped[int]        = mapped_column(Integer, primary_key=True)
    nickname:              Mapped[str]        = mapped_column(String(50), nullable=False)
    birth_date:            Mapped[date]       = mapped_column(Date, nullable=False)
    region_code:           Mapped[str | None] = mapped_column(String(10))
    interested_categories: Mapped[str | None] = mapped_column(String(100))
    education_level_code:  Mapped[str | None] = mapped_column(String(20))
    major_field_code:      Mapped[str | None] = mapped_column(String(20))
    job_status_code:       Mapped[str | None] = mapped_column(String(20))
    marital_status_code:   Mapped[str | None] = mapped_column(String(20))
    special_biz_code:      Mapped[str | None] = mapped_column(String(100))
    annual_income:         Mapped[int | None] = mapped_column(Integer)
