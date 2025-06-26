from enum import Enum


class OpenAIModel(Enum):
    """
    Enum for OpenAI judge models.
    """

    GPT_4o_MINI = "gpt-4o-mini"
    GPT_4o = "gpt-4o"
    gpt_4_1_nano = "gpt-4.1-nano"
    gpt_4_1_mini = "gpt-4.1-mini"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
