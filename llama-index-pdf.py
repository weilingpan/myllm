import logging
from datetime import datetime
logging.basicConfig(
    # filename=f'{datetime.now()}.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(message)s', 
    datefmt="%Y-%m-%d %H:%M:%S")

from llama_index.core import Settings
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
# from langchain.prompts import PromptTemplate

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.llms.openai import OpenAI
# from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


from llama_index.core.schema import (
    MetadataMode,
    NodeWithScore,
    TextNode,
)

from llama_index.core.response_synthesizers import (
    ResponseMode,
    get_response_synthesizer,
)

from typing import Union, List
from llama_index.core.node_parser import SentenceSplitter

import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.core.response.pprint_utils import pprint_response

from env import OPENAI_API_KEY as CUSTOM_OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = CUSTOM_OPENAI_API_KEY

top_k = 10
llm_model = "gpt-4o" #"gpt-4o-mini"
# o1-preview, o1-preview-2024-09-12, o1-mini, o1-mini-2024-09-12, gpt-4, gpt-4-32k, gpt-4-1106-preview, gpt-4-0125-preview, gpt-4-turbo-preview, gpt-4-vision-preview, gpt-4-1106-vision-preview, gpt-4-turbo-2024-04-09, gpt-4-turbo, gpt-4o, gpt-4o-2024-05-13, gpt-4o-2024-08-06, gpt-4o-2024-11-20, chatgpt-4o-latest, gpt-4o-mini, gpt-4o-mini-2024-07-18, gpt-4-0613, gpt-4-32k-0613, gpt-4-0314, gpt-4-32k-0314, gpt-3.5-turbo, gpt-3.5-turbo-16k, gpt-3.5-turbo-0125, gpt-3.5-turbo-1106, gpt-3.5-turbo-0613, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0301, text-davinci-003, text-davinci-002, gpt-3.5-turbo-instruct, text-ada-001, text-babbage-001, text-curie-001, ada, babbage, curie, davinci, gpt-35-turbo-16k, gpt-35-turbo, gpt-35-turbo-0125, gpt-35-turbo-1106, gpt-35-turbo-0613, gpt-35-turbo-16k-0613
embedding_model = "text-embedding-3-large" #OpenAI"text-embedding-ada-002" text-embedding-3-large text-embedding-3-small
logging.info(f"top_k={top_k}, llm_model={llm_model}, embedding_model={embedding_model}")

DEFAULT_CITATION_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000")) 
DEFAULT_CITATION_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "1000"))
logging.info(f"chunk_size={DEFAULT_CITATION_CHUNK_SIZE}, overlay={DEFAULT_CITATION_CHUNK_OVERLAP}")

# Set Model info
# Settings.llm = OpenAI(model=llm_model)
# Settings.embed_model = OpenAIEmbedding(model=embedding_model)

# EMBEDDING_AZURE_OPENAI_API_KEY = "xxxx"
# EMBEDDING_AZURE_DEPLOYMENT = "aiassistant-ada-002"
# EMBEDDING_AZURE_ENDPOINT = "https://xxxx.openai.azure.com"
# AZURE_OPENAI_API_VERSION =  "2024-06-01"
# azure_llm = AzureOpenAI(
#     model="gpt-4-1106-preview",
#     deployment_name=EMBEDDING_AZURE_DEPLOYMENT,
#     api_key=EMBEDDING_AZURE_OPENAI_API_KEY,
#     azure_endpoint=EMBEDDING_AZURE_DEPLOYMENT,
#     api_version=AZURE_OPENAI_API_VERSION,
# )
# azure_embedding = AzureOpenAIEmbedding(
#     model="text-embedding-ada-002",
#     api_key=EMBEDDING_AZURE_OPENAI_API_KEY,
#     azure_endpoint=EMBEDDING_AZURE_ENDPOINT,
#     deployment_name=EMBEDDING_AZURE_DEPLOYMENT,
#     api_version=AZURE_OPENAI_API_VERSION,
# )
# Settings.llm = azure_llm
# Settings.embed_model = azure_embedding

# llm = azure_llm







# 1. Define events
# Event: retrieved nodes to the create citations
class RetrieverEvent(Event):
    nodes: list[NodeWithScore]

# Event: citation nodes to the synthesizer
class CreateCitationsEvent(Event):
    nodes: list[NodeWithScore]


# 2. Citation Prompt Templates
# CITATION_QA_TEMPLATE = PromptTemplate(
#     "Please answer the question with citation based solely on the provided sources."
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "For every sentence you write, cite the citation number as [id]."
#     "And indicate the source of the file name of source."
#     "which supports pdf, txt, doc, excel, etc."
#     "Only cite a source when you are explicitly referencing it. "
#     "At the end of your answer: "
#     "Create a reference list of file names for each file you cited."
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1 (file_name=1.txt):\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2 (file_name=2.txt):\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1]."
#     "reference:\n"
#     "- [1] 1.txt\n"
#     "- [2] 2.txt\n"
#     "Now it's your turn. Below are several numbered sources of information:"
#     "\n------\n"
#     "{context_str}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

# CITATION_REFINE_TEMPLATE = PromptTemplate(
#     "Please answer the question with citation based solely on the provided sources."
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "For every sentence you write, cite the citation number as [id]."
#     "And indicate the source of the file name of source."
#     "which supports pdf, txt, doc, excel, etc."
#     "At the end of your answer: "
#     "Create a reference list of file names for each file you cited."
#     "At the end of your answer: Create a sources list of file names, and a link for each file you cited."
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1 (file_name=1.txt):\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2 (file_name=2.txt):\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1]."
#     "reference:\n"
#     "- [1] 1.txt\n"
#     "- [2] 2.txt\n"
#     "Now it's your turn. "
#     "We have provided an existing answer: {existing_answer}"
#     "Below are several numbered sources of information. "
#     "Use them to refine the existing answer. "
#     "If the provided sources are not helpful, you will repeat the existing answer."
#     "\nBegin refining!"
#     "\n------\n"
#     "{context_msg}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

# CITATION_QA_TEMPLATE = PromptTemplate(
#     "Please provide an answer based solely on the provided sources. "
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "Every answer should include at least one source citation. "
#     "And indicate the source of the file name."
#     "and file supports pdf, txt, doc, excel, etc."
#     "Only cite a source when you are explicitly referencing it. "
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1 (file_name=1.txt):\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2 (file_name=2.txt):\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1]."
#     "reference:"
#     "- [1] file_name\n"
#     "- [2] file_name\n"
#     "Now it's your turn. Below are several numbered sources of information:"
#     "\n------\n"
#     "{context_str}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

# CITATION_REFINE_TEMPLATE = PromptTemplate(
#     "Please provide an answer based solely on the provided sources. "
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "Every answer should include at least one source citation. "
#     "And indicate the source of the file name."
#     "and file supports pdf, txt, doc, excel, etc."
#     "Only cite a source when you are explicitly referencing it. "
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1 (file_name=1.txt):\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2 (file_name=1.txt):\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1]."
#     "reference:"
#     "- [1] file_name\n"
#     "- [2] file_name\n"
#     "Now it's your turn. "
#     "We have provided an existing answer: {existing_answer}"
#     "Below are several numbered sources of information. "
#     "Use them to refine the existing answer. "
#     "If the provided sources are not helpful, you will repeat the existing answer."
#     "\nBegin refining!"
#     "\n------\n"
#     "{context_msg}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

CITATION_QA_TEMPLATE = PromptTemplate(
    "You are a helpful assistant that provides answers based on the provided sources."
    "The system supports files in PDF, TXT, DOC, Excel, and other common formats, "
    "and source format is: [File Name].[File Extension]"
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "For every sentence in your response, include at least one source citation. "
    "Use the square brackets format e.g., [Number]."
    "Only cite a source when you are explicitly referencing it. "
    "At the end of the answer, summarize all the sources used in a 'reference list:'"
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "reference:\n"
    "[1] [File Name].[File Extension], char idx: [start_char_idx~end_char_idx], page: [page_label]\n"
    "[2] [File Name].[File Extension], char idx: [start_char_idx~end_char_idx], page: [page_label]\n"
    "Now it's your turn. Below are several numbered sources of information:"
    "\n------\n"
    "{context_str}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

CITATION_REFINE_TEMPLATE = PromptTemplate(
    "Please provide an answer based solely on the provided sources. "
    "When referencing information from a source, "
    "cite the appropriate source(s) using their corresponding numbers. "
    "For every sentence in your response, include at least one source citation. "
    "Use the square brackets format e.g., [Number]."
    "Only cite a source when you are explicitly referencing it. "
    "At the end of the answer, summarize all the sources used in a 'reference list:'"
    "If none of the sources are helpful, you should indicate that. "
    "For example:\n"
    "Source 1:\n"
    "The sky is red in the evening and blue in the morning.\n"
    "Source 2:\n"
    "Water is wet when the sky is red.\n"
    "Query: When is water wet?\n"
    "Answer: Water will be wet when the sky is red [2], "
    "which occurs in the evening [1].\n"
    "reference:\n"
    "[1] [File Name].[File Extension], char idx: [start_char_idx~end_char_idx], page: [page_label]\n"
    "[2] [File Name].[File Extension], char idx: [start_char_idx~end_char_idx], page: [page_label]\n"
    "Now it's your turn. "
    "We have provided an existing answer: {existing_answer}"
    "Below are several numbered sources of information. "
    "Use them to refine the existing answer. "
    "If the provided sources are not helpful, you will repeat the existing answer."
    "\nBegin refining!"
    "\n------\n"
    "{context_msg}"
    "\n------\n"
    "Query: {query_str}\n"
    "Answer: "
)

# 3. Workflow and Steps
class CitationQueryEngineWorkflow(Workflow):
    @step
    async def retrieve(
        self, ctx: Context, ev: StartEvent
    ) -> Union[RetrieverEvent, None]:
        logging.info(f"[StartEvent]")
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        logging.info(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        if ev.index is None:
            logging.info("Index is empty, load some documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=top_k)
        # retriever = ev.index.as_retriever(retriever_mode='default')
        nodes = retriever.retrieve(query)
        logging.info(f"Retrieved {len(nodes)} nodes.")
        return RetrieverEvent(nodes=nodes)
    
    @step
    async def create_citation_nodes(
        self, ev: RetrieverEvent
    ) -> CreateCitationsEvent:
        """
        Modify retrieved nodes to create granular sources for citations.

        Takes a list of NodeWithScore objects and splits their content
        into smaller chunks, creating new NodeWithScore objects for each chunk.
        Each new node is labeled as a numbered source, allowing for more precise
        citation in query results.

        Args:
            nodes (List[NodeWithScore]): A list of NodeWithScore objects to be processed.

        Returns:
            List[NodeWithScore]: A new list of NodeWithScore objects, where each object
            represents a smaller chunk of the original nodes, labeled as a source.
        """
        logging.info(f"create_citation_nodes ...")
        nodes = ev.nodes

        new_nodes: List[NodeWithScore] = []

        text_splitter = SentenceSplitter(
            chunk_size=DEFAULT_CITATION_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CITATION_CHUNK_OVERLAP,
        )

        for node in nodes:
            text_chunks = text_splitter.split_text(
                node.node.get_content(metadata_mode=MetadataMode.NONE)
            )

            for text_chunk in text_chunks:
                # logging.info(node.__dict__)
                page = node.node.metadata.get('page_label', None)
                text = f"Source {len(new_nodes)+1} (page: {page}, start char {node.node.start_char_idx} ~ end char {node.node.end_char_idx}):\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )

                logging.info(f"cleaning input node {len(new_nodes)+1} ...")
                # print(node.metadata["page_label"])
                new_node.node.text = text
                new_node.metadata["file_path"] = node.node.metadata["file_name"]
                # logging.info(new_node.__dict__)
                new_nodes.append(new_node)
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(
        self, ctx: Context, ev: CreateCitationsEvent
    ) -> StopEvent:
        """Return a streaming response using the retrieved nodes."""
        llm = OpenAI(model=llm_model) #TODO
            
        query = await ctx.get("query", default=None)

        synthesizer = get_response_synthesizer(
            llm=llm,
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )

        response = await synthesizer.asynthesize(query, nodes=ev.nodes)
        return StopEvent(result=response)
    

async def main():
    # Create Index
    folder = "./data/sample"
    if not os.path.exists("./citation2"):
        logging.info(f"Downloading and indexing documents from {folder}")
        documents = SimpleDirectoryReader(folder).load_data()
        # logging.info(f"documents={documents}")
        logging.info(f"Indexing documents: {len(documents)}")
        # for doc in documents:
        #     doc_info = {
        #         "id": doc.id_,
        #         "file_path": doc.metadata.get("file_path"),
        #         "file_name": doc.metadata.get("file_name"),
        #     }
        #     logging.info(doc_info)
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model = OpenAIEmbedding(model=embedding_model)
        )
        index.storage_context.persist(persist_dir="./citation")
    else:
        logging.info(f"Loading index from storage")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./citation"),
        )

    # Run the Workflow!
    logging.info(f"Init workflow ...")
    workflow = CitationQueryEngineWorkflow(timeout=30, verbose=False)
    # custom_query = "What information do you have"
    custom_query = "人工智慧分級介紹"
    # custom_query = "神經網路機器翻譯系統是誰發明的？"
    # custom_query = "what is langchain?"
    # custom_query = "當連接器電源pin的電流分配不均, 可能是什麼原因?"

    result = await workflow.run(query=custom_query, index=index)
    logging.info(f"\nResult:\n{result}\n")
    # logging.info(result.__dict__)
    
    # logging.info(f"============ Read Line ============")
    # def iter_lines(response):
    #     for line in response.split('\n'):
    #         yield line

    # for line in iter_lines(str(result)):
    #     logging.info(line)

    # # Check the citations.
    logging.info(f"============ Inspect ============")
    for idx, sn in enumerate(result.source_nodes):
        logging.info(f"============ Node {idx} ============")
        node_info = {
            "id": result.source_nodes[idx].id_,
            "file_name": result.source_nodes[idx].metadata.get("file_name"),
            "start_char_idx": result.source_nodes[idx].node.start_char_idx, #Start char index of the node.
            "end_char_idx": result.source_nodes[idx].node.end_char_idx, #End char index of the node.
            "score": result.source_nodes[idx].score,
            "doc_text": result.source_nodes[idx].node.get_text(),
            "page_label": result.source_nodes[idx].metadata.get("page_label", None)
        }
        logging.info(node_info)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
