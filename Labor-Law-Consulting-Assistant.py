# -*- coding: utf-8 -*-
import json
import time
from pathlib import Path
from typing import List, Dict
import re
import chromadb
import streamlit as st
from llama_index.core import VectorStoreIndex, StorageContext, Settings, get_response_synthesizer
from llama_index.core.schema import TextNode
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import PromptTemplate
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.llms.openai_like import OpenAILike

# ================== Streamlit页面配置 ==================
st.set_page_config(
    page_title="智能劳动法咨询助手",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="auto"
)


def disable_streamlit_watcher():
    """Patch Streamlit to disable file watcher"""

    def _on_script_changed(_):
        return

    from streamlit import runtime
    runtime.get_instance()._on_script_changed = _on_script_changed


# ================== 配置类 ==================
class Config:
    EMBED_MODEL_PATH = r"/root/MyPython/demo_15/embedding_model/sungw111/text2vec-base-chinese-sentence"
    RERANK_MODEL_PATH = r"/root/model/BAAI/bge-reranker-large/BAAI/bge-reranker-large"

    DATA_DIR = "./data"
    VECTOR_DB_DIR = "./chroma_db"
    PERSIST_DIR = "./storage"

    COLLECTION_NAME = "chinese_labor_laws"
    TOP_K = 10
    RERANK_TOP_K = 3


# ================== 缓存资源初始化 ==================
@st.cache_resource(show_spinner="初始化模型中...")
def init_models():
    embed_model = HuggingFaceEmbedding(
        model_name=Config.EMBED_MODEL_PATH,
    )

    # llm = OpenAILike(
    #     model="/home/cw/llms/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #     api_base="http://localhost:23333/v1",
    #     api_key="fake",
    #     context_window=4096,
    #     is_chat_model=True,
    #     is_function_calling_model=False,
    # )
    llm = OpenAILike(
        model="glm-4",  # 可选模型：glm-4, glm-3-turbo, characterglm等
        api_base="https://open.bigmodel.cn/api/paas/v4",  # 关键！必须指定此端点
        api_key="f56e497a962b45739347c45bf40c1372.AyiwcVRUF6Hiy2dz",
        context_window=128000,  # 按需调整（glm-4实际支持128K）
        is_chat_model=True,
        is_function_calling_model=False,  # GLM暂不支持函数调用
        max_tokens=1024,  # 最大生成token数（按需调整）
        temperature=0.3,  # 推荐范围 0.1~1.0
        top_p=0.7  # 推荐范围 0.5~1.0
    )

    reranker = SentenceTransformerRerank(
        model=Config.RERANK_MODEL_PATH,
        top_n=Config.RERANK_TOP_K
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    return embed_model, llm, reranker


@st.cache_resource(show_spinner="加载知识库中...")
def init_vector_store(_nodes):
    chroma_client = chromadb.PersistentClient(path=Config.VECTOR_DB_DIR)
    chroma_collection = chroma_client.get_or_create_collection(
        name=Config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    storage_context = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
    )

    if chroma_collection.count() == 0 and _nodes is not None:
        storage_context.docstore.add_documents(_nodes)
        index = VectorStoreIndex(
            _nodes,
            storage_context=storage_context,
            show_progress=True
        )
        storage_context.persist(persist_dir=Config.PERSIST_DIR)
        index.storage_context.persist(persist_dir=Config.PERSIST_DIR)
    else:
        storage_context = StorageContext.from_defaults(
            persist_dir=Config.PERSIST_DIR,
            vector_store=ChromaVectorStore(chroma_collection=chroma_collection)
        )
        index = VectorStoreIndex.from_vector_store(
            storage_context.vector_store,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
    return index


# ================== 数据处理 ==================
def load_and_validate_json_files(data_dir: str) -> List[Dict]:
    """加载并验证JSON法律文件"""
    json_files = list(Path(data_dir).glob("*.json"))
    assert json_files, f"未找到JSON文件于 {data_dir}"

    all_data = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                # 验证数据结构
                if not isinstance(data, list):
                    raise ValueError(f"文件 {json_file.name} 根元素应为列表")
                for item in data:
                    if not isinstance(item, dict):
                        raise ValueError(f"文件 {json_file.name} 包含非字典元素")
                    for k, v in item.items():
                        if not isinstance(v, str):
                            raise ValueError(f"文件 {json_file.name} 中键 '{k}' 的值不是字符串")
                all_data.extend({
                                    "content": item,
                                    "metadata": {"source": json_file.name}
                                } for item in data)
            except Exception as e:
                raise RuntimeError(f"加载文件 {json_file} 失败: {str(e)}")

    print(f"成功加载 {len(all_data)} 个法律文件条目")
    return all_data


def create_nodes(raw_data: List[Dict]) -> List[TextNode]:
    """添加ID稳定性保障"""
    nodes = []
    for entry in raw_data:
        law_dict = entry["content"]
        source_file = entry["metadata"]["source"]

        for full_title, content in law_dict.items():
            # 生成稳定ID（避免重复）
            node_id = f"{source_file}::{full_title}"

            parts = full_title.split(" ", 1)
            law_name = parts[0] if len(parts) > 0 else "未知法律"
            article = parts[1] if len(parts) > 1 else "未知条款"

            node = TextNode(
                text=content,
                id_=node_id,  # 显式设置稳定ID
                metadata={
                    "law_name": law_name,
                    "article": article,
                    "full_title": full_title,
                    "source_file": source_file,
                    "content_type": "legal_article"
                }
            )
            nodes.append(node)

    print(f"生成 {len(nodes)} 个文本节点（ID示例：{nodes[0].id_}）")
    return nodes


# ================== 界面组件 ==================
def init_chat_interface():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg.get("cleaned", msg["content"])  # 优先使用清理后的内容

        with st.chat_message(role):
            st.markdown(content)

            # 如果是助手消息且包含思维链
            if role == "assistant" and msg.get("think"):
                with st.expander("📝 模型思考过程（历史对话）"):
                    for think_content in msg["think"]:
                        st.markdown(f'<span style="color: #808080">{think_content.strip()}</span>',
                                    unsafe_allow_html=True)

            # 如果是助手消息且有参考依据（需要保持原有参考依据逻辑）
            if role == "assistant" and "reference_nodes" in msg:
                show_reference_details(msg["reference_nodes"])


def show_reference_details(nodes):
    with st.expander("查看支持依据"):
        for idx, node in enumerate(nodes, 1):
            meta = node.node.metadata
            st.markdown(f"**[{idx}] {meta['full_title']}**")
            st.caption(f"来源文件：{meta['source_file']} | 法律名称：{meta['law_name']}")
            st.markdown(f"相关度：`{node.score:.4f}`")
            # st.info(f"{node.node.text[:300]}...")
            st.info(f"{node.node.text}")


# ================== 主程序 ==================
def main():
    # 禁用 Streamlit 文件热重载
    disable_streamlit_watcher()
    st.title("⚖️ 智能劳动法咨询助手")
    st.markdown("欢迎使用劳动法智能咨询系统，请输入您的问题，我们将基于最新劳动法律法规为您解答。")

    # 初始化会话状态
    if "history" not in st.session_state:
        st.session_state.history = []

    # 加载模型和索引
    embed_model, llm, reranker = init_models()

    # 初始化数据
    if not Path(Config.VECTOR_DB_DIR).exists():
        with st.spinner("正在构建知识库..."):
            raw_data = load_and_validate_json_files(Config.DATA_DIR)
            nodes = create_nodes(raw_data)
    else:
        nodes = None

    index = init_vector_store(nodes)
    retriever = index.as_retriever(similarity_top_k=Config.TOP_K, vector_store_query_mode="hybrid", alpha=0.5)
    # 调整检索器配置（扩大召回范围并启用混合检索）
    # retriever = index.as_retriever(
    #     similarity_top_k=20,  # 从10提升到20
    #     vector_store_query_mode="hybrid",  # 混合检索模式
    #     alpha=0.5,  # 平衡密集检索与稀疏检索
    #     filters={"content_type": "legal_article"}  # 添加元数据过滤
    # )
    response_synthesizer = get_response_synthesizer(verbose=True)

    # 聊天界面
    init_chat_interface()

    if prompt := st.chat_input("请输入劳动法相关问题"):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 处理查询
        with st.spinner("正在分析问题..."):
            start_time = time.time()

            # 检索流程
            initial_nodes = retriever.retrieve(prompt)
            reranked_nodes = reranker.postprocess_nodes(initial_nodes, query_str=prompt)

            # 过滤节点
            MIN_RERANK_SCORE = 0.4
            filtered_nodes = [node for node in reranked_nodes if node.score > MIN_RERANK_SCORE]

            if not filtered_nodes:
                response_text = "⚠️ 未找到相关法律条文，请尝试调整问题描述或咨询专业律师。"
            else:
                # 生成回答
                response = response_synthesizer.synthesize(prompt, nodes=filtered_nodes)
                response_text = response.response

            # 显示回答
            with st.chat_message("assistant"):
                # 提取思维链内容并清理响应文本
                think_contents = re.findall(r'<think>(.*?)</think>', response_text, re.DOTALL)
                cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()

                # 显示清理后的回答
                st.markdown(cleaned_response)

                # 如果有思维链内容则显示
                if think_contents:
                    with st.expander("📝 模型思考过程（点击展开）"):
                        for content in think_contents:
                            st.markdown(f'<span style="color: #808080">{content.strip()}</span>',
                                        unsafe_allow_html=True)

                # 显示参考依据（保持原有逻辑）
                show_reference_details(filtered_nodes[:3])

            # 添加助手消息到历史（需要存储原始响应）
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,  # 保留原始响应
                "cleaned": cleaned_response,  # 存储清理后的文本
                "think": think_contents  # 存储思维链内容
            })


if __name__ == "__main__":
    main()