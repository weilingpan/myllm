from llama_index.core import Settings
from llama_index.core.workflow import Event
from llama_index.core.schema import NodeWithScore
from llama_index.core.prompts import PromptTemplate
# from langchain.prompts import PromptTemplate
# from llama_index.llms.replicate import Replicate

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

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

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer
from llama_index.core.response.pprint_utils import pprint_response

# 1. Define events
# Event: retrieved nodes to the create citations
class RetrieverEvent(Event):
    nodes: list[NodeWithScore]

# Event: citation nodes to the synthesizer
class CreateCitationsEvent(Event):
    nodes: list[NodeWithScore]


# 2. Citation Prompt Templates
DEFAULT_CITATION_CHUNK_SIZE = 2000
DEFAULT_CITATION_CHUNK_OVERLAP = 1000

# CITATION_QA_TEMPLATE = PromptTemplate(
#     "You are a helpful assistant that provides answers based on the provided sources."
#     "The system supports files in PDF, TXT, DOC, Excel, and other common formats."
#     "The source format is: [File Name].[File Extension]"
#     "Please provide an answer based solely on the provided sources. "
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "For every sentence in your response, include at least one source document and page number in parentheses."
#     "Use the square brackets format e.g., [1]."
#     "Only cite a source when you are explicitly referencing it."
#     "At the end of the answer, summarize all the sources used in a 'Source List:'"
#     "Use the format:"
#     "Source List:"
#     "- [id] [File Name].[File Extension], Page: [Page Number]\n"
#     "If none of the sources are helpful, you should indicate that."
#     "For example:\n"
#     "Source 1:\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2:\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1]."
#     "reference:"
#     "- [1] [File Name].[File Extension], Page: [Page Number]\n"
#     "- [2] [File Name].[File Extension], Page: [Page Number]\n"
#     "Now it's your turn. Below are several numbered sources of information:"
#     "\n------\n"
#     "{context_str}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

# CITATION_REFINE_TEMPLATE = PromptTemplate(
#     "The initial answer below needs to be refined for accuracy and completeness."
#     "You are refining the assistant's response to ensure:"
#     "Ensure every sentence includes a citation in the square brackets format: e.g., [1]."
#     "A final 'Source List' is included with all referenced sources."
#     "Use the format:"
#     "Source List:\n"
#     "- [id] [File Name].[File Extension], Page: [Page Number]\n"
    
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
    "[1] [File Name].[File Extension], Page: [Page Number]\n"
    "[2] [File Name].[File Extension], Page: [Page Number]\n"
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
    "[1] [File Name].[File Extension], Page: [Page Number]\n"
    "[2] [File Name].[File Extension], Page: [Page Number]\n"
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
        print(f"[StartEvent]")
        "Entry point for RAG, triggered by a StartEvent with `query`."
        query = ev.get("query")
        if not query:
            return None

        print(f"Query the database with: {query}")

        # store the query in the global context
        await ctx.set("query", query)

        if ev.index is None:
            print("Index is empty, load some documents before querying!")
            return None

        retriever = ev.index.as_retriever(similarity_top_k=3) #score_threshold=.6
        nodes = retriever.retrieve(query)
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
        print(f"create_citation_nodes ...")
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
                text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )
                
                print(f"cleaning input ...")
                new_node.node.text = text
                new_node.metadata["file_path"] = node.node.metadata["file_name"]
                print(new_node.__dict__)
                new_nodes.append(new_node)
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(
        self, ctx: Context, ev: CreateCitationsEvent
    ) -> StopEvent:
        """Return a streaming response using the retrieved nodes."""
        llm = OpenAI(model=llm_model)
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
    
llm_model = "gpt-4o-mini"
embedding_model = "text-embedding-3-small"
async def main():
    # Set Model info
    
    os.environ["OPENAI_API_KEY"] = "None"

    Settings.llm = OpenAI(model=llm_model,temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=embedding_model) #"text-embedding-3-small"
    # Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model) #"BAAI/bge-small-en-v1.5"

    # # set tokenizer to match LLM
    # Settings.tokenizer = AutoTokenizer.from_pretrained(
    #     "NousResearch/Llama-2-7b-chat-hf"
    # )

    # Create Index
    if not os.path.exists("./citation2"):
        print(f"Downloading and indexing documents")
        documents = SimpleDirectoryReader("./data/sample").load_data()
        # print(f"documents={documents}")
        print(f"Indexing documents: {len(documents)}")
        for doc in documents:
            doc_info = {
                "id": doc.id_,
                "file_path": doc.metadata.get("file_path"),
                "file_name": doc.metadata.get("file_name"),
            }
            print(doc_info)
        index = VectorStoreIndex.from_documents(
            documents,
        )
        index.storage_context.persist(persist_dir="./citation")
        print('ref_docs ingested: ', len(index.ref_doc_info))
    else:
        print(f"Loading index from storage")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./citation"),
        )

    # query_input = "人工智慧分級介紹"
    # response = index.as_query_engine().query(query_input)
    # pprint_response(response, show_source=True)

    # Run the Workflow!
    print(f"Init workflow ...")
    workflow = CitationQueryEngineWorkflow()
    # custom_query = "What information do you have"
    custom_query = "人工智慧分級介紹"
    # custom_query = "神經網路機器翻譯系統是誰發明的？"
    result = await workflow.run(query=custom_query, index=index)
    print(f"\nResult:\n{result}\n")
    # print(result.__dict__)

    # # Check the citations.
    # print(f"============ Inspect ============")
    # for idx, sn in enumerate(result.source_nodes):
    #     print(f"============ Node {idx} ============")
    #     node_info = {
    #         "id": result.source_nodes[idx].id_,
    #         "file_name": result.source_nodes[idx].metadata.get("file_name"),
    #         "start_char_idx": result.source_nodes[idx].node.start_char_idx,
    #         "end_char_idx": result.source_nodes[idx].node.end_char_idx,
    #         "score": result.source_nodes[idx].score,
    #         "doc_text": result.source_nodes[idx].node.get_text()
    #     }
    #     print(node_info)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
