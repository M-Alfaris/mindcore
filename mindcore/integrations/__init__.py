"""Framework integrations for Mindcore.

Provides adapters for popular AI frameworks, updated for 2025 patterns:

LangChain (v0.3.x):
- MindcoreChatMessageHistory: Implements BaseChatMessageHistory for use with
  RunnableWithMessageHistory and LangGraph. (Recommended for 2025+)
- MindcoreMemory: Legacy memory interface for backwards compatibility.

CrewAI (2025):
- MindcoreRAGStorage: Implements RAGStorage interface for use as custom
  storage backend with ShortTermMemory, EntityMemory, etc. (Recommended)
- MindcoreCrewMemory: Legacy memory interface for backwards compatibility.

LlamaIndex (2025):
- MindcoreMemoryBlock: Implements BaseMemoryBlock for use with the new
  Memory class and agent memory system. (Recommended)
- MindcoreIndexMemory: Legacy memory interface for backwards compatibility.

Example (LangChain v0.3.x - Modern):
    from langchain_core.runnables.history import RunnableWithMessageHistory
    from mindcore.integrations import MindcoreChatMessageHistory

    def get_session_history(session_id: str):
        return MindcoreChatMessageHistory(
            storage="postgresql://localhost/mindcore",
            session_id=session_id,
            user_id="user_123",
        )

    chain = RunnableWithMessageHistory(llm, get_session_history)

Example (CrewAI 2025 - Modern):
    from crewai import Crew
    from crewai.memory import ShortTermMemory
    from mindcore.integrations import MindcoreRAGStorage

    storage = MindcoreRAGStorage(
        storage="postgresql://localhost/mindcore",
        storage_type="short_term",
    )

    crew = Crew(
        agents=[...],
        memory=True,
        short_term_memory=ShortTermMemory(storage=storage),
    )

Example (LlamaIndex 2025 - Modern):
    from llama_index.core.memory import Memory
    from mindcore.integrations import MindcoreMemoryBlock

    memory = Memory.from_defaults(
        memory_blocks=[
            MindcoreMemoryBlock(
                storage="postgresql://localhost/mindcore",
                user_id="user_123",
            ),
        ],
    )
"""

# LangChain integrations
# CrewAI integrations
from .crewai import MindcoreCrewMemory, MindcoreRAGStorage
from .langchain import MindcoreChatMessageHistory, MindcoreMemory

# LlamaIndex integrations
from .llamaindex import MindcoreIndexMemory, MindcoreMemoryBlock


__all__ = [
    "MindcoreChatMessageHistory",
    "MindcoreCrewMemory",
    "MindcoreIndexMemory",
    "MindcoreMemory",
    "MindcoreMemoryBlock",
    "MindcoreRAGStorage",
]
