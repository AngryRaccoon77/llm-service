from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import LlamaCpp
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_core.exceptions import OutputParserException
import requests
import os

# Путь к модели GGUF
model_path = "D:/model/aovchinnikov/T-lite-it-1.0-Q4_K_M-GGUF/t-lite-it-1.0-q4_k_m.gguf"

# Инициализация LlamaCpp
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=14,
    n_batch=512,
    n_ctx=2048,
    f16_kv=True,
    verbose=True
)

# Получение API-ключа из переменных окружения
API_KEY = "5c505f2649e1d6117484321b4f107816"
if not API_KEY:
    raise ValueError("API-ключ OpenWeather не найден. Установите переменную окружения OPENWEATHER_API_KEY.")

# Функция для получения погоды
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
    except requests.exceptions.RequestException as e:
        return f"Ошибка при получении данных о погоде: {str(e)}"


def search_db(query: str, query_ngql: str = None) -> str:
    """
    Отправляет запрос к сервису базы данных через HTTP POST
    """
    service_url = "http://localhost:8081/search"

    try:
        # Отправляем POST-запрос с параметрами
        response = requests.post(
            service_url,
            json={
                "query": query,
                "query_ngql": query_ngql
            },
            timeout=10  # Таймаут 10 секунд
        )

        # Проверяем статус код
        response.raise_for_status()

        # Парсим JSON ответ
        result = response.json()

        # Проверяем наличие ожидаемых данных в ответе
        if "result" in result:
            return result["result"]
        else:
            return "Получен неожиданный формат ответа от сервиса"

    except requests.exceptions.RequestException as e:
        # Обрабатываем ошибки соединения
        return f"Ошибка соединения с сервисом: {str(e)}"
    except Exception as e:
        # Общие ошибки
        return f"Ошибка при обработке запроса: {str(e)}"

# Создание инструмента
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Получает текущую погоду в указанном городе с использованием API OpenWeather."
)

search_db_tool = Tool(
    name="search_db",
    func=search_db,
    description="Поиск в векторной базе данных qdrant и knowledge graph nebula с помощью ngql."
)

# Инициализация агента
agent = initialize_agent(
    tools=[weather_tool, search_db_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Функция для взаимодействия с агентом
def ask_question(question):
    try:
        response = agent.run(question)
    except OutputParserException as e:
        try:
            response = str(e).split("Could not parse LLM output: `")[1].split("`")[0]
        except IndexError:
            response = "Извините, не удалось обработать ваш запрос."
    return response

# FastAPI приложение
app = FastAPI()

class Question(BaseModel):
    question: str

@app.post("/ask")
async def ask(question: Question):
    answer = ask_question(question.question)
    return {"answer": answer}
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)