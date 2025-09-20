from typing import Literal, Optional
import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

Provider = Literal["ollama", "openai"]


def get_llm(provider: Provider = "ollama", model: Optional[str] = None):
    """Return a LangChain Chat model for the chosen provider.

    - ollama: runs locally via Ollama daemon (default model: gemma2)
    - openai: uses OpenAI API (requires OPENAI_API_KEY). Default model: gpt-4o-mini
    """
    if provider == "ollama":
        name = model or os.environ.get("OLLAMA_MODEL", "gemma2")
        return ChatOllama(model=name, temperature=0.1)
    elif provider == "openai":
        # Requires env OPENAI_API_KEY
        name = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=name, temperature=0.1)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
