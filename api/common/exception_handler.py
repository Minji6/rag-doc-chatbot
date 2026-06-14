import logging
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Request

logger = logging.getLogger(__name__)

###############################################################
# 예외 처리기
###############################################################
# HTTPException이 발생했을때 처리하는 함수
async def http_exception_handler(request: Request, e: HTTPException):
    return JSONResponse({
        "message": "HTTP 에러 처리함",
        "detail": str(e)
    })

# 유효성 검사 예외가 발생했을때 처리하는 함수
async def validation_exception_handler(request: Request, e: RequestValidationError):
    return JSONResponse({
        "message": "요청 데이터 유효성 검사 실패 처리함",
        "detail": e.errors()
    })

# 그 이외에 발생하는 모든 예외를 처리하는 함수
async def exception_handler(request: Request, e: Exception):
    return JSONResponse({
        "message": "에러 처리함",
        "detail": str(e)
    })
    
###############################################################
# 예외 처리기 일괄 등록 함수
###############################################################
def register_exception_handler(app: FastAPI):
    app.exception_handler(404)(http_exception_handler)
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)
    app.exception_handler(Exception)(exception_handler)