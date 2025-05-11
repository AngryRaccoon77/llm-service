from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import LlamaCpp
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.exceptions import OutputParserException
import requests
import os
import logging

# === Настройка логирования ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === Путь к модели GGUF ===
model_path = "/Users/enterprise/Downloads/t-lite-it-1.0-q4_k_m.gguf"

# === Инициализация LLM ===
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=14,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    verbose=True
)

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


# === Функция поиска в Qdrant через RAG сервис ===
def search_db(query: str, query_ngql: str = None) -> str:
    service_url = "http://localhost:8086/rag/process"
    try:
        response = requests.post(
            service_url,
            json={"query": query},
            timeout=10
        )
        response.raise_for_status()

        # Парсим JSON и получаем результат
        data = response.json()
        return data.get("result", "")

    except Exception as e:
        logger.error(f"Ошибка при обращении к RAG-сервису: {e}")
        return f"Ошибка при работе с RAG-сервисом: {str(e)}"

# === Инструменты ===
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Получает текущую погоду в указанном городе."
)

search_db_tool = Tool(
    name="search_db",
    func=search_db,
    description="Поиск в векторной базе данных qdrant с помощью RAG сервиса."
)

# === Агент ===
agent = initialize_agent(
    tools=[weather_tool, search_db_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# === Модель для входных данных ===
class QuestionRequest(BaseModel):
    question: str


# === FastAPI ===
app = FastAPI()

class RagRequest(BaseModel):
    query: str


@app.post("/rag-process")
def rag_process(request: RagRequest):
    try:
        # Запускаем агент с RAG инструментом
        answer = agent.run(request.query)
        return {"response": answer}
    except OutputParserException as e:
        try:
            answer = str(e).split("Could not parse LLM output: `")[1].split("`")[0]
            return {"response": answer}
        except IndexError:
            return {"error": "Извините, не удалось обработать ваш запрос."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/ask")
async def ask(request: QuestionRequest):
    try:
        answer = agent.run(request.question)
        return {"answer": answer}
    except OutputParserException as e:
        logger.warning(f"OutputParserException: {e}")
        try:
            answer = str(e).split("Could not parse LLM output: `")[1].split("`")[0]
            return {"answer": answer}
        except IndexError:
            return {"error": "Извините, не удалось обработать ваш запрос."}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === Точка входа ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8084)