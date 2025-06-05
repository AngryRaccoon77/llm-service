# application/services.py
import logging
from langchain.agents import initialize_agent, AgentType
from langchain_core.exceptions import OutputParserException

from llmService.app.infrastructure.llm_service import llm_service
from llmService.app.infrastructure.memory_service import memory_service
from llmService.app.infrastructure.tool_services import get_tools
from llmService.app.infrastructure import config

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self):
        self.llm = llm_service.get_llm()
        self.tools = get_tools()

    def process_chat(self, chat_id: str, query: str) -> str:
        try:
            memory = memory_service.get_memory(chat_id)

            agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                memory=memory,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                verbose=True,
                agent_kwargs={"prefix": config.SYSTEM_PROMPT}
            )

            response = agent.run(query)
            return response

        except OutputParserException as e:
            logger.warning(f"OutputParserException: {e}")
            try:
                # Try to extract the answer from the exception string
                answer = str(e).split("Could not parse LLM output: ")[1]
                return answer
            except IndexError:
                return "Не удалось разобрать вывод агента."
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            # Re-raise or handle as a specific application error
            raise e


# Singleton instance
chat_service = ChatService()