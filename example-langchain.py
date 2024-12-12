# python version: 3.10.12
# langchain version: 0.2.12


import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["OPENAI_API_KEY"] = "None"


model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="hi!"),
]

result = model.invoke(messages)
answer = result.content
print(f"AI result: {answer}")
print(result)
