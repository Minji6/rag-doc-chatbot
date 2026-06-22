from datetime import date, datetime
from sqlalchemy import Integer, String, Date, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from api.common.sqlalchemy_conf import Base


class User(Base):
    __tablename__ = "users"

    user_id:      Mapped[int]            = mapped_column(Integer, primary_key=True)
    nickname:     Mapped[str]            = mapped_column(String(50), nullable=False)
    birth_date:   Mapped[date]           = mapped_column(Date, nullable=False)
    zipcd:        Mapped[str | None]     = mapped_column(String(50))
    category:     Mapped[str | None]     = mapped_column(String(100))
    schoolcd:     Mapped[str | None]     = mapped_column(String(50))
    plcymajorcd:  Mapped[str | None]     = mapped_column(String(50))
    jobcd:        Mapped[str | None]     = mapped_column(String(50))
    mrgsttscd:    Mapped[str | None]     = mapped_column(String(50))
    sbizcd:       Mapped[str | None]     = mapped_column(String(200))
    earncndsecd:  Mapped[int | None]     = mapped_column(Integer)
    created_at:   Mapped[datetime]       = mapped_column(DateTime, server_default=func.now())
    updated_at:   Mapped[datetime]       = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
