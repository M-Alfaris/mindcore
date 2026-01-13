"""Framework integrations for Mindcore.

Provides adapters for popular AI frameworks:
- LangChain: MindcoreMemory
- CrewAI: MindcoreCrewMemory
- LlamaIndex: MindcoreIndexMemory

Example:
    # LangChain
    from mindcore.integrations import MindcoreMemory
    memory = MindcoreMemory(storage="postgresql://localhost/mindcore")
    chain = ConversationChain(llm=llm, memory=memory)

    # CrewAI
    from mindcore.integrations import MindcoreCrewMemory
    memory = MindcoreCrewMemory(storage="postgresql://localhost/mindcore")
    crew = Crew(agents=[...], memory=memory)

    # LlamaIndex
    from mindcore.integrations import MindcoreIndexMemory
    memory = MindcoreIndexMemory(storage="postgresql://localhost/mindcore")
    chat_engine = index.as_chat_engine(memory=memory)
"""

from .crewai import MindcoreCrewMemory
from .langchain import MindcoreMemory
from .llamaindex import MindcoreIndexMemory


__all__ = [
    "MindcoreCrewMemory",
    "MindcoreIndexMemory",
    "MindcoreMemory",
]
