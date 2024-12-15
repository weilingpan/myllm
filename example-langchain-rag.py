# reference: https://medium.com/@cch.chichieh/rag%E5%AF%A6%E4%BD%9C%E6%95%99%E5%AD%B8-langchain-llama2-%E5%89%B5%E9%80%A0%E4%BD%A0%E7%9A%84%E5%80%8B%E4%BA%BAllm-d6838febf8c4
# reference: https://medium.com/jimmy-wang/langchain-rag%E5%AF%A6%E6%88%B0%E7%AC%AC%E4%B8%80%E7%AB%99-efe975f4c3bd

import os

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain import hub

from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


from env import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 1. Load PDF
print(f"===> Loading PDF...")
loader = PyMuPDFLoader("data/sample/ai.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")
# pdf會分page，每個page會是一個Document
# [Document(
#     metadata={
#         'source': 'data/sample/ai.pdf', 
#         'file_path': 'data/sample/ai.pdf', 
#         'page': 0, 
#         'total_pages': 3, 
#         'format': 'PDF 1.5', 
#         'title': '',
#         'author': '', 
#         'subject': '', 
#         'keywords': '', 
#         'creator': 'Adobe InDesign CS6 (Windows)', 
#         'producer': 'Adobe PDF Library 10.0.1', 
#         'creationDate': "D:20210322091016+08'00'", 
#         'modDate': "D:20210322091018+08'00'", 
#         'trapped': ''}, 
#     page_content="",
#     ...
# dict_keys(['id', 'metadata', 'page_content', 'type'])


# 2. Split text
print(f"===> Splitting text...")
# 將文件或文字分割成一個個 chunk
# Chunk 是指將一段資料分成固定大小的部分。每一塊通常大小一致，但最後一塊可能會有例外（取決於資料總量和chunk的大小）。
# 如果是pdf, 因為已經有page了, 所以chunk也是根據page來切, 不會跨page
chunk_size = 1000
chunk_overlap = 500
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
splits = text_splitter.split_documents(docs)
print(f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
print(f"Created {len(splits)} splits(chunks) from {len(docs)} pages")
#TODO: 加入start_index, end_index
# for s in splits:
#     print(s.__dict__)
    # print(s.metadata)

# 3. Create Embeddings
# 將分割後的的 chunk 文字轉換為向量
print(f"===> Creating embeddings...")
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# model_kwargs = {'device': 'cpu'}
# embeddings = HuggingFaceEmbeddings(model_name=model_name,
#                                   model_kwargs=model_kwargs)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = OpenAIEmbeddings()


# 4. Create Vector Store
# 常見的 VectorDB 有 Chroma、Pinecone、FAISS等
print(f"===> Creating vector store...")
persist_directory = 'db'
vectordb = Chroma.from_documents(
    documents=splits, #chunks
    embedding=embeddings, 
    persist_directory=persist_directory)

# 5. Setup LLM
print(f"===> Setup LLM with prompt...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# system_template = "You are a helpful assistant."
# prompt_template = ChatPromptTemplate.from_messages(
#     [("system", system_template), 
#      ("user", "{text}")]
# )
# prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Answer in Italian:"""
# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template=prompt_template,
# )

prompt = hub.pull("rlm/rag-prompt")
print(f"Prompt:\n{prompt}")

# llm_chain = prompt | llm
# question = "人工智慧的分級?"
# llm_chain.invoke({"question": question})


# 6. Create retriever
print(f"===> Create retriever...")
retriever = vectordb.as_retriever()

# 7. Integrate Retrieval with QA
print(f"===> Integrate Retrieval with QA...")
# qa = RetrievalQA.from_chain_type(
#     llm=llm_chain, 
#     chain_type="stuff", 
#     retriever=retriever, 
#     verbose=True
# )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("===> 完成建立 RAG chain")

query = "人工智慧的分級介紹?"
print(f"===> Question: {query}")
result = rag_chain.invoke(query)
# print(result)
print(result.__dict__)

# result = qa.invoke(query)
# print(f"===> Answer: {result}")