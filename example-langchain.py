# python version: 3.10.12
# langchain version: 0.2.12


import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "xxxx"

parser = StrOutputParser()
model = ChatOpenAI(model="gpt-4")

system_template = "You are a helpful assistant, Translate the following into {language}."
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), 
     ("user", "{text}")]
)
print(f"Prompt:\n{prompt_template}")

#### 測試prompt_template ####
# result = prompt_template.invoke(
#     {"language": "italian",
#      "text": "hi"})
# messages = result.to_messages()
# print(f"messages={messages}")
#### 測試prompt_template ####

# messages = [
#     SystemMessage(content="You are a helpful assistant"),
#     HumanMessage(content="hi!"),
# ]

chain = prompt_template | model | parser
answer = chain.invoke(
    {"language": "italian", 
     "text": "hi"}) # 加variable到prompt template

# chain = model | parser # the chain has two steps: first the language model is called, then the result of that is passed to the output parser.
# answer = chain.invoke(messages) 
print(f"AI result: {answer}")

# result = model.invoke(messages)
# answer = result.content
# print(f"AI result: {answer}")
# print(result)