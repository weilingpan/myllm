# 版本
# langchain                          0.2.12
# langchain-community                0.2.11
# langchain-core                     0.2.43
# langchain-openai                   0.1.20
# langchain-text-splitters           0.2.4
# langfuse                           2.45.2
# langsmith                          0.1.147

# new version
# langchain                               0.3.11
# langchain-community                     0.3.11
# langchain-core                          0.3.24
# langchain-ollama                        0.2.1
# langchain-openai                        0.2.12
# langchain-text-splitters                0.3.2


import os

from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_core.documents import Document

from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import (
    HuggingFaceHubEmbeddings as community_HuggingFaceHubEmbeddings,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


from langchain_community.vectorstores import FAISS




from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.chains import create_citation_fuzzy_match_chain


os.environ["OPENAI_API_KEY"] = "None"

def highlight(text, span):
    return (
        "..."
        + text[span[0] - 20 : span[0]]
        + "*"
        + "\033[91m"
        + text[span[0] : span[1]]
        + "\033[0m"
        + "*"
        + text[span[1] : span[1] + 20]
        + "..."
    )


def run():
    question = "What did the author do during college?"
    context = """
    My name is Jason Liu, and I grew up in Toronto Canada but I was born in China.
    I went to an arts highschool but in university I studied Computational Mathematics and physics. 
    As part of coop I worked at many companies including Stitchfix, Facebook.
    I also started the Data Science club at the University of Waterloo and I was the president of the club for 2 years.
    """


    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = create_citation_fuzzy_match_chain(llm)
    result = chain.run(question=question, context=context)
    # result = chain.invoke({"input_documents": docs, "question": question})
    print(f"\nResult:\n{result}\n")

    for fact in result.answer:
        print("Statement:", fact.fact)
        for span in fact.get_spans(context):
            print("Citation:", highlight(context, span))
        print()


if __name__ == "__main__":
    print(f"Run langchain flow ...")
    run()
