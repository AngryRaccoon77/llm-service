# infrastructure/tool_services.py
import requests
import logging
from langchain.tools import Tool
from . import config
from .llm_service import llm_service

logger = logging.getLogger(__name__)


def get_weather(city: str) -> str:
    """Gets the current weather in a specified city."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": config.OPENWEATHER_API_KEY,
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


def search_db(query: str, query_ngql: str = None) -> str:
    """Searches the university database for any information related to the university, including study programs."""
    try:
        response = requests.post(
            config.RAG_SERVICE_URL,
            json={"query": query},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        retriever_response = data.get("result", "")

        llm = llm_service.get_llm()
        if llm.get_num_tokens(retriever_response) > 1800:
            prompt = f"Суммаризируй следующий текст в нескольких предложениях: {retriever_response}"
            retriever_response = llm(prompt, max_tokens=200)
        return retriever_response
    except Exception as e:
        logger.error(f"Error while communicating with RAG service: {e}")
        return f"Error while working with RAG service: {str(e)}"


def get_tools() -> list:
    """Initializes and returns the list of agent tools."""
    weather_tool = Tool(
        name="get_weather",
        func=get_weather,
        description="Получает текущую погоду в указанном городе."
    )

    search_db_tool = Tool(
        name="search_db",
        func=search_db,
        description="Поиск в базе данных университета, для поиска любой информации связанной с университетом, в том числе и программы обучения."
    )

    return [weather_tool, search_db_tool]