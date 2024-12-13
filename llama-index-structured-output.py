"""
測 .pdf 有 page 資訊
再微調 citation node
調整workflow timeout
有加logfile,需要的話可以儲存成file
效果看起來還不錯！

可以用內網
"""

# ContextRelevancyEvaluator???

# Citation Query Engine similar to RAG + Reranking, the notebook focuses on how to implement intermediate steps in between retrieval and generation. A good example of how to use the Context object in a workflow.

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
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding


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

from pydantic import BaseModel, Field
from typing import List


os.environ["OPENAI_API_KEY"] = "None"

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


# 1. Define events
# Event: retrieved nodes to the create citations
class RetrieverEvent(Event):
    nodes: list[NodeWithScore]

# Event: citation nodes to the synthesizer
class CreateCitationsEvent(Event):
    nodes: list[NodeWithScore]


class Output(BaseModel):
    """Output containing the response, page numbers, and confidence."""

    response: str = Field(..., description="The answer to the question.")
    page_numbers: List[int] = Field(
        ...,
        description="The page numbers of the sources used to answer this question. Do not include a page number if the context is irrelevant.",
    )
    confidence: float = Field(
        ...,
        description="Confidence value between 0-1 of the correctness of the result.",
    )
    confidence_explanation: str = Field(
        ..., description="Explanation for the confidence score"
    )
    citations: list = Field(
        ..., description="List of citations used to answer the question."
    )


# 2. Citation Prompt Templates
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
        # Working with Global Context/State#
        await ctx.set("query", query)

        if ev.index is None:
            logging.info("Index is empty, load some documents before querying!")
            return None

        # build retriever
        retriever = ev.index.as_retriever(similarity_top_k=top_k)
        # retriever = ev.index.as_retriever(retriever_mode='default')

        #TODO retrieve from milvus
        # from db.db_milvus import vectorstore
        # doc_list = ["675a5cd988e47f1031ba72b5"]
        # retrieval_result = (
        #     vectorstore("RAG_Documents_test")
        #         .as_retriever(
        #             search_kwargs={
        #                 "k": 10,
        #                 "expr": f"doc_id in {doc_list}",
        #                 "score_threshold": 0.4,
        #             },
        #             search_type="similarity_score_threshold",
        #         )
        #         ._get_relevant_documents(query=query, run_manager=None)
        # )
        # [
        # Document(
        #     metadata={'pk': 452356727992503968, 
        #                 'sparse_vector': {0: 0.0}, 
        #                 'source': '/llmdocuments/regina_pan/人工智慧的原理與應用.pdf', 
        #                 'doc_id': '6757942b3ae2e63d3cf2d681', 
        #                 'filename': '人工智慧的原理與應用.pdf', 
        #                 'page': -1, 
        #                 'label': 'summary', 
        #                 'metadata': {}}, 
        #                 page_content='[Summary for 人工智慧的原理與應用.pdf] \n人工智慧的發展可以分為四個級別，包括自動控制、探索推論、機器學習和深度學習。這些技術被應用在各個領域，包括翻譯、影像識別、農業、醫療、金融、安全和客服等。人工智慧可以幫助進行植物病蟲害分析、糖尿病視網膜病變診斷、頭部頸部癌症診斷治療、風險控管、金融監理、安全防護、精準行銷和犯罪行為評估等。同時，人工智慧也可以用於情緒表情分析和智慧客服，提高生活質量和工作效率。'),
        # Document(
        #     metadata={'pk': 452356727992503957, 
        #                 'sparse_vector': {0: 0.0}, 
        #                 'source': '/llmdocuments/regina_pan/人工智慧的原理與應用.pdf', 
        #                 'doc_id': '6757942b3ae2e63d3cf2d681', 
        #                 'filename': '人工智慧的原理與應用.pdf', 
        #                 'page': 1, 
        #                 'label': 'text', 
        #                 'metadata': {}}, 
        #                 page_content='[人工智慧的原理與應用.pdf] \n人工智慧的分級與歷史\n人工智慧(AI：Artificial Intelligence)，一\n個吸引人們卻又令大家害怕的名詞，吸引我們\n的是一個會思考可以協助我們處理工作的智慧\n型機器人，可以替我們帶小孩洗衣做飯；讓我\n們害怕的是這個機器人自己會思考，哪天他不\n聽話了怎麼辦？更慘的是，哪天老闆發現他比\n我還好用，那我不就失業了？許多人以為人工\n智慧就是科幻電影裡會思考的機器人，人工智\n慧真的這麼神奇嗎？現在的人工智慧到底發展\n到什麼程度了？它到底有那些限制呢？\n人工智慧的定義與範圍\n人工智慧(AI：Artificial Intelligence) 是指\n人類製造出來的機器所表現出來的智慧，人工\n智慧討論研究的範圍很廣，包括：演繹、推理\n和解決問題、知識表示法、規劃與學習、自\n然語言處理、機器感知、機器社交、創造力\n等，而我們常常聽到的「機器學習(Machine \nlearning)」是屬於人工智慧的一部分，另外\n「深度學習(Deep learning)」又屬於機器學習\n的一部分，如圖一所示。\n人工智慧的分級\n人工智慧的依照機器（電腦）能夠處理\n與判斷的能力區分為四個分級如下：\n→第一級人工智慧(First level AI)：自動控制\n第一級人工智慧是指機器（電腦）含有\n人工智慧的原理與應用\n台北福星曙光衛星社 曲建仲博士Hightech\n圖一  人工智慧、機器學習、深度學習的範圍\n參考資料：blogs.nvidia.com.tw\n85\n臺灣扶輪\u30002021.4'), 
        # ]
                



        nodes = retriever.retrieve(query)
        logging.info(f"The number of Nodes: {len(nodes)}")
        # print(nodes)
        # [NodeWithScore(
        #     node=TextNode(
        #         id_='8083eb7d-8ab7-4b3b-b475-2fd78354c5de', 
        #         embedding=None, 
        #         metadata={
        #             'file_path': '/home/regina/Desktop/Regina/myllm/data/sample/sample.doc', 
        #             'file_name': 'sample.doc', 
        #             'file_type': 'application/msword', 
        #             'file_size': 4477, 
        #             'creation_date': '2024-12-11', 
        #             'last_modified_date': '2024-12-11'}, 
        #         excluded_embed_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], 
        #         excluded_llm_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], 
        #         relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='e3d05b5c-221a-459c-b27b-9448e50e6bfa', node_type='4', metadata={'file_path': '/home/regina/Desktop/Regina/myllm/data/sample/sample.doc', 'file_name': 'sample.doc', 'file_type': 'application/msword', 'file_size': 4477, 'creation_date': '2024-12-11', 'last_modified_date': '2024-12-11'}, hash='dd6bd4b025a35bbe84ad7a97ed9139057a2f2220f5597838816186d8b7a2bf58'), <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='27688fab-b800-4551-b5be-73f781d9fbfa', node_type='1', metadata={}, hash='f5ab0e716f27dc3ee74840f4c684fb41aeaa6ed4386b54562865e87e373ca829')}, 
        #         metadata_template='{key}: {value}', 
        #         metadata_separator='\n', 
        #         text='內文xxxx', 
        #         mimetype='text/plain', 
        #         start_char_idx=0, 
        #         end_char_idx=825, 
        #         metadata_seperator='\n', 
        #         text_template='{metadata_str}\n\n{content}'), 
        #         score=0.6855801038531206)
        # ]
        # logging.info(f"===== Retrieved {len(nodes)} nodes. =====")
        # for idx, n in enumerate(nodes):
        #     logging.info(f"=============== Node {idx} ===============")
        #     logging.info(f"{n.__dict__}")
            # {'node': TextNode(id_='8d2001bf-825b-4177-b1cb-f76a4043c011', 
            #                   embedding=None, 
            #                   metadata={
            #                       'file_path': '/home/regina/Desktop/Regina/myllm/data/sample/ai.txt', 
            #                       'file_name': 'ai.txt', 
            #                       'file_type': 'text/plain', 
            #                       'file_size': 4393,
            #                       'creation_date': '2024-12-10', 
            #                       'last_modified_date': '2024-12-10'}, 
            #                   excluded_embed_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], 
            #                   excluded_llm_metadata_keys=['file_name', 'file_type', 'file_size', 'creation_date', 'last_modified_date', 'last_accessed_date'], 
            #                   relationships={<NodeRelationship.SOURCE: '1'>: RelatedNodeInfo(node_id='8f4d649f-8c95-48c2-a5bc-5009d1151e6a', 
            #                                                                                  node_type='4', 
            #                                                                                  metadata={'file_path': '/home/regina/Desktop/Regina/myllm/data/sample/ai.txt', 'file_name': 'ai.txt', 'file_type': 'text/plain', 'file_size': 4393, 'creation_date': '2024-12-10', 'last_modified_date': '2024-12-10'}, hash='fbe4d21c5c08035b1dee7e56cf69efeda84d0c7a7ea3daee792a9911e6d1b2d2'), 
            #                                 <NodeRelationship.PREVIOUS: '2'>: RelatedNodeInfo(node_id='13a137f6-6fe7-45d4-99c0-a65eb0eb6f16', node_type='1', metadata={'file_path': '/home/regina/Desktop/Regina/myllm/data/sample/ai.txt', 'file_name': 'ai.txt', 'file_type': 'text/plain', 'file_size': 4393, 'creation_date': '2024-12-10', 'last_modified_date': '2024-12-10'}, hash='d7918a505372e7319e854df6db622c44d4875aa4441fca7a539938313764428a'), 
            #                                 <NodeRelationship.NEXT: '3'>: RelatedNodeInfo(node_id='85257e89-e37f-4ff5-91ac-a088c65dc70c', node_type='1', metadata={}, hash='efda943718174ee68b0e3dd1e0f17308856d225284e9fbb87d45dae207d46bc2')}, 
            #                   metadata_template='{key}: {value}', 
            #                   metadata_separator='\n', 
            #                   text='頭部頸部癌症訪斷治療：Deepmind 與英\n國倫敦大學醫學院合作，利用機器學習參與治\n療方案的設計過程，協助醫護人員分辨癌變組\n織和健康組織，細分過程由4 小時縮短到1 小\n時，同時提高了放射治療的效率。\n交易與理財諮詢(Robo advisor)：投資理\n財機器人可以依照客戶不同的財務目標、風險\n容忍度、投資範圍等演算出建議的資產配置，\n系統自動將資金配置投資於幾個指數型基金，\n過程只需要10 分鐘。\n風險控管模型建構(Risk control)：目前銀\n行的信用評分制度多依賴聯合徵信中心的信用\n相關資料作為評分參數，透過人工智慧與大數\n據可以分析客戶的票證支付、電信公司、公用\n事業、大賣場、購物商城、社群網站資料發展\n出全方位的信用評分系統。\n金融監理科技(RegTech：Regulation\nTechnology)：金融科技興起造成金融監理及法\n令遵循的管控工作日益繁瑣，人工智慧可以將\n行員與客戶之間的交談錄音及錄影資料，透過\n特定的關鍵字檢索，定期進行過濾與檢視，能\n快速地確認其中是否有違反相關的作業規定。\n安全防護身分辨識(Identification)：經由\n人工智慧演算法進行臉部、聲紋、虹膜、靜\n脈、指紋等生物辨識，作為客戶進行金融交易\n的主要方式，時間更短、精確度更高，例如：\n支付寶在2015 年時就以機器視覺與深度學習\n研發「人臉支付」技術。\n精準行銷(Precision marketing)：利用人工\n智慧可以分析購買行為、客戶特徵、社群行為\n等，了解各種客戶特性，並且利用數據開發預\n測模型，作為處理信用額度、風險管理、產品\n訂價的工具，依據不同預測結果提供客製化\n服務。', 
            #                   mimetype='text/plain', 
            #                   start_char_idx=605, 
            #                   end_char_idx=1313, 
            #                   metadata_seperator='\n', 
            #                   text_template='{metadata_str}\n\n{content}'), 
            #                   'score': 0.48572812760303574}
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
                # text = f"Source {len(new_nodes)+1} (page: {page}, start char {node.node.start_char_idx} ~ end char {node.node.end_char_idx}):\n{text_chunk}\n"
                text = f"Source {len(new_nodes)+1}:\n{text_chunk}\n"

                new_node = NodeWithScore(
                    node=TextNode.model_validate(node.node), score=node.score
                )

                logging.info(f"cleaning input node {len(new_nodes)+1} ...")
                new_node.node.text = text
                new_node.metadata["start_char_idx"] = node.node.start_char_idx
                new_node.metadata["end_char_idx"] = node.node.end_char_idx
                # new_node.metadata["file_path"] = node.node.metadata["file_name"]
                logging.info(f"metadata:\n{new_node.metadata}")
                # logging.info(new_node.__dict__)
                new_nodes.append(new_node)
        return CreateCitationsEvent(nodes=new_nodes)

    @step
    async def synthesize(
        self, ctx: Context, ev: CreateCitationsEvent
    ) -> StopEvent:
        """Return a streaming response using the retrieved nodes."""
        llm = OpenAI(model=llm_model) #TODO
        sllm = llm.as_structured_llm(output_cls=Output)

        query = await ctx.get("query", default=None)

        response_synthesizer = get_response_synthesizer(
            llm=sllm,
            text_qa_template=CITATION_QA_TEMPLATE,
            refine_template=CITATION_REFINE_TEMPLATE,
            response_mode=ResponseMode.COMPACT,
            use_async=True,
        )

        response = await response_synthesizer.asynthesize(query, nodes=ev.nodes) #非同步
        # await ctx.set("answer", response)
        return StopEvent(result=response)

# LlamaIndex provides a variety of modules enabling LLMs to produce outputs in a structured format. 


async def main():
    # Create Index
    folder = "./data/elec"
    # folder = "./data/sample"
    if not os.path.exists("./citation2"):
        logging.info(f"Downloading and indexing documents from {folder}")
        # Automatically select the best file reader given file extensions.
        # uses a in-memory `SimpleVectorStore`
        reader = SimpleDirectoryReader(folder)
        documents = reader.load_data()

        logging.info(f"Indexing documents: {len(documents)}") #如果是pdf, 會分割page
        for d in documents:
            logging.info(f"load documents(before embedding) ...........................")
            logging.info(d.__dict__)

        # build index
        index = VectorStoreIndex.from_documents(
            documents,
            embed_model = OpenAIEmbedding(model=embedding_model)
        )
        # from llama_index.core.schema import TextNode
        # node1 = TextNode(text="<text_chunk>", id_="<node_id>")
        # node2 = TextNode(text="<text_chunk>", id_="<node_id>")
        # nodes = [node1, node2]
        # index = VectorStoreIndex(nodes)

        index.storage_context.persist(persist_dir="./citation")
    else:
        logging.info(f"Loading index from storage")
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir="./citation"),
        )

    # Run the Workflow!
    logging.info(f"Init workflow ...")
    workflow = CitationQueryEngineWorkflow(timeout=120, verbose=False)
    # custom_query = "What information do you have"
    # custom_query = "人工智慧分級介紹"
    # custom_query = "神經網路機器翻譯系統是誰發明的？"
    # custom_query = "what is langchain?"
    # custom_query = "電源 Via 的配置方式對 Via 電流分佈的影響"
    custom_query = "當連接器電源pin的電流分配不均, 可能是什麼原因?"


    # If you want to maintain state across multiple runs of a workflow, 
    # you can pass a previous context into the .run() method.
    handler = workflow.run(query=custom_query, index=index)
    result = await handler
    logging.info(f"\nResult:\n{result}\n")
    print(len(result.citations))
    # logging.info(f"\nResponse:\n{result.response}\n")
    # print(str(result.response))
    # logging.info(result.__dict__)

    # # continue with next run
    # handler = workflow.run(query=custom_query, index=index, ctx=handler.ctx)
    # result = await handler
    # logging.info(f"\nResult:\n{result}\n")
    
    # logging.info(f"============ Read Line ============")
    # def iter_lines(response):
    #     for line in response.split('\n'):
    #         yield line

    # for line in iter_lines(str(result)):
    #     logging.info(line)

    # # Check the citations.
    # logging.info(f"============ Inspect ============")
    # for idx, sn in enumerate(result.source_nodes):
    #     logging.info(f"============ Node {idx} ============")
    #     node_info = {
    #         "id": result.source_nodes[idx].id_,
    #         "file_name": result.source_nodes[idx].metadata.get("file_name"),
    #         "start_char_idx": result.source_nodes[idx].node.start_char_idx,
    #         "end_char_idx": result.source_nodes[idx].node.end_char_idx,
    #         "score": result.source_nodes[idx].score,
    #         "doc_text": result.source_nodes[idx].node.get_text(),
    #         "page_label": result.source_nodes[idx].metadata.get("page_label", None)
    #     }
    #     logging.info(node_info)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
