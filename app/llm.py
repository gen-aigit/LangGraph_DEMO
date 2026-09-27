"""LangChain OpenAI chat model shared by the agent nodes."""

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


def get_llm() -> BaseChatModel:
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
