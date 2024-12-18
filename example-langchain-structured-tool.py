import os
os.environ["OPENAI_API_KEY"] = "None"

from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState



# 1. Define Structured Output
class WeatherResponse(BaseModel):
    """Respond to the user with this"""

    temperature: float = Field(description="The temperature in fahrenheit")
    wind_directon: str = Field(
        description="The direction of the wind in abbreviated form"
    )
    wind_speed: float = Field(description="The speed of the wind in km/h")


# 2. Final structured response from the agent
# Inherit 'messages' key from MessagesState, which is a list of chat messages
class AgentState(MessagesState):
    final_response: WeatherResponse

# 3. Define tool
@tool
def get_weather(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "It is cloudy in NYC, with 5 mph winds in the North-East direction and a temperature of 70 degrees"
    elif city == "sf":
        return "It is 75 degrees and sunny in SF, with 3 mph winds in the South-East direction"
    else:
        raise AssertionError("Unknown city")


# 4. Bind output as tool
tools = [get_weather, WeatherResponse]

model = ChatOpenAI(model="gpt-4o-mini")
model_with_tools = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(WeatherResponse)


# structured_llm = prompt | model_with_structured_output
structured_llm = model_with_structured_output
result = structured_llm.invoke("what's something funny about woodpeckers")
print(result)
