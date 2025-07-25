import os
import asyncio
from openai import OpenAI
from litellm import (
    completion, 
    acompletion, 
    batch_completion_models_all_responses,
    batch_completion
)
from langchain_openai import ChatOpenAI

from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

openai_api_key = OPENAI_API_KEY
model_name = "gpt-4.1-nano"

"""ChatOpenAI"""
# ### invoke
# model = ChatOpenAI(model=model_name, streaming=False)
# response = model.invoke("hello")
# print(response)


# ### streaming
# model = ChatOpenAI(model=model_name, streaming=True)
# response = model.stream("hello")
# for chunk in response:
#     print(chunk)


# ### batch
# model = ChatOpenAI(model=model_name, streaming=False)
# response = model.batch(["hello", "how are you?"])
# print(f"Batch response: {len(response)} items")
# for i, resp in enumerate(response):
#     print(f"Response {i+1}: {resp.content}")
# # print(response)
"""
[
    AIMessage(
        content='Hello! How can I assist you today?', 
        additional_kwargs={'refusal': None}, 
        response_metadata={
            'token_usage': 
                {
                    'completion_tokens': 9, 
                    'prompt_tokens': 8, 
                    'total_tokens': 17, 
                    'completion_tokens_details': {
                        'accepted_prediction_tokens': 0, 
                        'audio_tokens': 0, 
                        'reasoning_tokens': 0, 
                        'rejected_prediction_tokens': 0
                    }, 
                    'prompt_tokens_details': {
                        'audio_tokens': 0, 
                        'cached_tokens': 0
                    }
                }, 
            'model_name': 'gpt-4.1-nano-2025-04-14', 
            'system_fingerprint': None, 
            'finish_reason': 'stop', 
            'logprobs': None
        }, 
        id='run--9ef20aea-c910-4ce7-877d-d8ae37e8c699-0', 
        usage_metadata={
            'input_tokens': 8,
            'output_tokens': 9,
            'total_tokens': 17,
            'input_token_details': {'audio': 0, 'cache_read': 0}, 
            'output_token_details': {'audio': 0, 'reasoning': 0}}
    ), 
    AIMessage(content="I'm doing well, thank you! How can I assist you today?", additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 11, 'total_tokens': 25, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-nano-2025-04-14', 'system_fingerprint': None, 'finish_reason': 'stop', 'logprobs': None}, id='run--19a307d4-488f-4f7a-81f9-8b780be6d14c-0', usage_metadata={'input_tokens': 11, 'output_tokens': 14, 'total_tokens': 25, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})]
"""


"""OpenAI Client"""
# client = OpenAI(
#     api_key=OPENAI_API_KEY,
# )

# models = client.models.list()
# for model in models.data:
#     print(f"Model ID: {model.id}")


""" OpenAI Chat Completions API """
# completion = client.chat.completions.create(
#     model=model_name,
#     messages=[
#         {
#             "role": "user",
#             "content": "hello",
#         }
#     ],
#     stream=False,
#     max_tokens=100
# )

# print(completion.choices) #[0].message.content


""" OpenAI Completions API """
# ## batch
# completion = client.completions.create(
#     # 不是所有model都支援 v1/completions (model endpoint compatibility reference: https://platform.openai.com/docs/models)
#     model="gpt-3.5-turbo-instruct",
#     prompt=["what is python", "how are you?"],
#     # n=2,
#     stream=False,
#     # logprobs=3,
#     max_tokens=50
# )

# print(f"Total choices: {len(completion.choices)}")
# for idx, result in enumerate(completion.choices):
#     print(f"\n=============== 第 {idx} 個結果 ===============")
#     print(result.text)




""" litellm """
# messages = [
#     {
#         "content": "Hello, how are you?",
#         "role": "user"
#     }
# ]

# response = completion(model=f"openai/{model_name}", messages=messages)
# print(response)

# ##### Async 
# async def test_get_response():
#     user_message = "Hello, how are you?"
#     messages = [{"content": user_message, "role": "user"}]
#     response = await acompletion(model=f"openai/{model_name}", messages=messages)
#     return response

# response = asyncio.run(test_get_response())
# print(response)


# ##### streaming
# response = completion(model=f"openai/{model_name}", messages=messages, stream=True)
# for part in response:
#     print(part.choices[0].delta.content or "")



##### batch
### completion call to many models: Return All Responses
# responses = batch_completion_models_all_responses(
#     models=[f"openai/{model_name}", "openai/gpt-4.1-mini"], 
#     messages=[{"role": "user", "content": "hello"}]
# )
# print(responses)


client = OpenAI(api_key="sk-1234", base_url="http://localhost:4000")
response = client.chat.completions.create(
    model="gpt-4.1-nano,gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "hello"}
    ],
)
print(response)


### Send multiple completion calls to 1 model
# responses = batch_completion(
#     model=model_name,
#     messages = [
#         [
#             {
#                 "role": "user",
#                 "content": "good morning? "
#             }
#         ],
#         [
#             {
#                 "role": "user",
#                 "content": "what's the time? "
#             }
#         ]
#     ]
# )
# print(f"Total responses: {len(responses)}")
# for idx, response in enumerate(responses):
#     print(f"\n=============== 第 {idx} 個結果 ===============")
#     # print(response)
#     print(response.choices[0].message.content)  