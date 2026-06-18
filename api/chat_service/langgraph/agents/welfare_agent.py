from ._stub_agent import StubAgent

class WelfareAgent(StubAgent):
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        super().__init__(domain_label="복지문화")