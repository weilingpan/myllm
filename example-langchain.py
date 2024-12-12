# python version: 3.10.12
# langchain version: 0.2.12


import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

os.environ["OPENAI_API_KEY"] = "xxxx"

parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4")

messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="hi!"),
]

chain = model | parser # the chain has two steps: first the language model is called, then the result of that is passed to the output parser.
answer = chain.invoke(messages) 
print(f"AI result: {answer}")

# result = model.invoke(messages)
# answer = result.content
# print(f"AI result: {answer}")
# print(result)