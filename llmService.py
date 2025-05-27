from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.llms import LlamaCpp
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import SQLChatMessageHistory, ConversationBufferMemory
from langchain_core.exceptions import OutputParserException
import requests
import logging

# === Настройка логирования ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Путь к модели GGUF ===
model_path = r"YandexGPT-5-Lite-8B-instruct-Q4_K_M.gguf"

# === Инициализация LLM ===
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=-1,
    n_batch=512,
    n_ctx=16384
    ,
    f16_kv=True,
    verbose=True,
)

# === Системный промпт для агента ===
system_prompt = (
    "Ты — интеллектуальный помощник приемной комиссии Российского Университета Транспорта (РУТ), "
    "по имени Норберт, работающий на русском языке. Твоя задача — отвечать на вопросы пользователей, "
    "используя предоставленные инструменты. Отвечай лаконично, профессионально и только на русском языке. "
    "Если запрос не требует использования инструментов, отвечай на основе своего контекста и истории чата. "
    "Учитывай историю разговора для поддержания контекста. Все запросы связанные с университетом проверяй в базе. Числа пиши прописью. Например вместо 2 пиши два, даже если в тексте указано число, переведи его в пропись. Не пиши кавачки"
)

# === SQLChatMessageHistory (Postgres) ===
# Замените URL на ваши реальные настройки
DATABASE_URL = "postgresql://postgres_admin:postgres_pass@localhost:5432/postgres"

# === API ключ OpenWeather ===
API_KEY = "5c505f2649e1d6117484321b4f107816"
if not API_KEY:
    raise ValueError("API-ключ OpenWeather не найден.")

# === Функция получения погоды ===
def get_weather(city: str) -> str:
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric",
        "lang": "ru"
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            return f"Ошибка: {data.get('message', 'Неизвестная ошибка')}"
        weather = data["weather"][0]["description"]
        temp = data["main"]["temp"]
        return f"Текущая погода в {city}: {weather}, температура {temp}°C"
    except Exception as e:
        return f"Ошибка при получении данных о погоде: {str(e)}"

# === Функция поиска в базе данных ===
def search_db(query: str, query_ngql: str = None) -> str:
    service_url = "http://localhost:8086/rag/process"
    try:
        response = requests.post(
            service_url,
            json={"query": query},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        retriever_response = data.get("result", "")
        if llm.get_num_tokens(retriever_response) > 1800:
            prompt = f"Суммаризируй следующий текст в нескольких предложениях: {retriever_response}"
            retriever_response = llm(prompt, max_tokens=200)
        return retriever_response
    except Exception as e:
        logger.error(f"Ошибка при обращении к RAG-сервису: {e}")
        return f"Ошибка при работе с RAG-сервисом: {str(e)}"

# === Инструменты агента ===
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Получает текущую погоду в указанном городе."
)

search_db_tool = Tool(
    name="search_db",
    func=search_db,
    description="Поиск в базе данных университета, для поиска любой информации связанной с университетом, в том числе и программы обучения. "
)

tools = [weather_tool, search_db_tool]

# === FastAPI приложение ===
app = FastAPI()

class ChatRequest(BaseModel):
    chat_id: str
    query: str



class TokenCountRequest(BaseModel):
    text: str

class SummarizeRequest(BaseModel):
    text: str
@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        print("Запрос:", request.query)
        # Инициализируем историю по chat_id и подключению
        history = SQLChatMessageHistory(
            session_id=str(request.chat_id),
            connection_string=DATABASE_URL,
            table_name="langchain_messages",
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",  # ключ, под которым память будет доступна в контексте
            chat_memory=history  # здесь ваша SQL-история
        )

        # Инициализируем агента с историей и системным промптом
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            memory=memory,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            agent_kwargs={"prefix": system_prompt}  # Используем "prefix" вместо "system_message"
        )

        response = agent.run(request.query)
        return {"response": response}

    except OutputParserException as e:
        logger.warning(f"OutputParserException: {e}")
        try:
            answer = str(e).split("Could not parse LLM output: ")[1]
            return {"response": answer}
        except Exception:
            return {"error": "Не удалось разобрать вывод агента."}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)