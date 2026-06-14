import logging

from langchain_core.callbacks import BaseCallbackHandler
import json

###############################################################
# 커스텀 콜백 핸들러 - LLM 호출 상세 정보 출력
###############################################################     
class LoggingCallbackHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.LoggingCallbackHandler")
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.logger.info("\n*******************************\n")
        self.logger.info("===== 요청 프롬프트 =====")
        self.logger.info(prompts)
        self.logger.info("\n===== 요청 추가 파라미터 =====")
        self.logger.info(json.dumps(kwargs, indent=2, ensure_ascii=False, default=str))

    def on_llm_end(self, response, **kwargs):
        self.logger.info("\n===== 응답 메시지 =====")
        self.logger.info(response)        
        self.logger.info("\n===== 응답 추가 파라미터 =====")
        self.logger.info(json.dumps(kwargs, indent=2, ensure_ascii=False, default=str))
        self.logger.info("\n*******************************\n")