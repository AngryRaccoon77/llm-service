from langchain.memory import SQLChatMessageHistory, ConversationBufferMemory
from . import config

class MemoryService:
    def get_memory(self, chat_id: str) -> ConversationBufferMemory:
        history = SQLChatMessageHistory(
            session_id=str(chat_id),
            connection_string=config.DATABASE_URL,
            table_name="langchain_messages",
        )
        return ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=history,
            return_messages=True
        )

memory_service = MemoryService()