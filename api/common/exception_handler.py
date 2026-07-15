import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request, status

logger = logging.getLogger(__name__)

###############################################################
# 예외 처리기
###############################################################
# HTTPException이 발생했을때 처리하는 함수
async def http_exception_handler(request: Request, e: HTTPException):
    return JSONResponse(
        status_code=e.status_code,
        content={"message": str(e.detail)}
    )

# 유효성 검사 예외가 발생했을때 처리하는 함수
async def validation_exception_handler(request: Request, e: RequestValidationError):
    errors = e.errors()
    if errors:
        first_error = errors[0]
        field = first_error.get("loc", ["field"])[-1]
        msg = first_error.get("msg", "요청 데이터 유효성 검사 실패")
        message = f"{field}: {msg}"
    else:
        message = "요청 데이터 유효성 검사 실패"

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"message": message}
    )

# 그 이외에 발생하는 모든 예외를 처리하는 함수
# 예외 내용은 서버 로그에만 남긴다 — str(e)를 응답에 담으면 DB 접속 정보·내부 경로 등이
# 클라이언트에 노출될 수 있다.
async def exception_handler(request: Request, e: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "서버 에러"}
    )
    
###############################################################
# 예외 처리기 일괄 등록 함수
###############################################################
def register_exception_handler(app: FastAPI):
    app.exception_handler(404)(http_exception_handler)
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.exception_handler(Exception)(exception_handler)