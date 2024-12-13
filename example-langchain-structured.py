import os
from langchain_openai import ChatOpenAI
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import Annotated, TypedDict

os.environ["OPENAI_API_KEY"] = "None"



# Pydantic
# class Joke(BaseModel):
#     """Joke to tell user."""

#     setup: str = Field(description="The setup of the joke")
#     punchline: str = Field(description="The punchline to the joke")
#     rating: Optional[int] = Field(
#         default=None, description="How funny the joke is, from 1 to 10"
#     )

# TypedDict
class Joke(TypedDict):
    """Joke to tell user."""

    setup: Annotated[str, ..., "The setup of the joke"]
    punchline: Annotated[str, ..., "The punchline of the joke"]
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]


llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Joke)
for chunk in structured_llm.stream("Tell me a joke about cats"):
    print(chunk)

# result = structured_llm.invoke("Tell me a joke about cats")
# print(result)
