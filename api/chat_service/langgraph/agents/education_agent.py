from ._stub_agent import StubAgent

class EducationAgent(StubAgent):
    def __init__(self, model: str = "openai:gpt-4o-mini") -> None:
        super().__init__(domain_label="교육")