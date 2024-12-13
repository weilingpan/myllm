import os
from langchain_openai import ChatOpenAI
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate


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


system = """You are a hilarious comedian. Your specialty is knock-knock jokes. \
Return a joke which has the setup (the response to "Who's there?") and the final punchline (the response to "<setup> who?").

Here are some examples of jokes:

example_user: Tell me a joke about planes
example_assistant: {{"setup": "Why don't planes ever get tired?", "punchline": "Because they have rest wings!", "rating": 2}}

example_user: Tell me another joke about planes
example_assistant: {{"setup": "Cargo", "punchline": "Cargo 'vroom vroom', but planes go 'zoom zoom'!", "rating": 10}}

example_user: Now about caterpillars
example_assistant: {{"setup": "Caterpillar", "punchline": "Caterpillar really slow, but watch me turn into a butterfly and steal the show!", "rating": 5}}"""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{input}")])

llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(Joke)

few_shot_structured_llm = prompt | structured_llm
result = few_shot_structured_llm.invoke("what's something funny about woodpeckers")
print(result)

# for chunk in structured_llm.stream("Tell me a joke about cats"):
#     print(chunk)

# result = structured_llm.invoke("Tell me a joke about cats")
# print(result)
