import logging

from rich.markdown import Markdown

from config import config

from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.text import Text

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

logger = logging.getLogger("rag")
console = Console()


# ---------------- Config ----------------
OLLAMA_URL = config["ollama_url"]
LLM_MODEL = config["model"]

EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "./chroma_db"


# ---------------- LLM ----------------
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    streaming=True,
)


# ---------------- Embeddings + DB ----------------
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)


# ---------------- Tool ----------------
@tool
def search_kb(query: str) -> str:
    """Search internal knowledge base for relevant context."""

    logger.info(f'TOOL CALL search_kb(query="{query}")')

    docs = retriever.invoke(query)

    logger.info(f"TOOL RESULT {len(docs)} chunks returned")

    output_chunks = []

    for i, d in enumerate(docs):

        logger.info(
            f"""
CHUNK {i + 1}
PAGE: {d.metadata.get('page')}
SECTION: {d.metadata.get('section', 'Unknown')}

{d.page_content[:100]}
"""
        )

        output_chunks.append(
            f"""
[Page {d.metadata.get('page')}]
[Section: {d.metadata.get('section', 'Unknown')}]

{d.page_content}
"""
        )

    return "\n\n".join(output_chunks)


tools = [search_kb]
llm_with_tools = llm.bind_tools(tools)


# ---------------- System Prompt ----------------
SYSTEM_PROMPT = SystemMessage(content="""
You are a precise RAG assistant.

Rules:
- Always use the knowledge base tool first when needed.
- If information required is not returned by the knowledge base, query the knowledge base more than once.
- Do not say the information is not present until you query the knowledge base at least 3 times.

- Prefer exact matches from the retrieved context.
- Use ALL relevant retrieved chunks before answering.
- Pay special attention to warranty exclusions, contaminant tables, warnings, and numbered procedures.
- If multiple retrieved chunks discuss the same topic, combine the information into a complete answer.

- ALWAYS use markup in your response.
- If info is missing, say it is not present.
- IMPORTANT: KEEP ANSWERS CONCISE.
- Return only the final answer

DO NOT RESPOND TO questions already answered.

ALWAYS generate a response.
""")


# ---------------- Agent Node ----------------
def call_model(state: MessagesState):
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------- Graph ----------------
graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "agent")

graph.add_conditional_edges("agent", tools_condition)

graph.add_edge("tools", "agent")

app = graph.compile()


# ---------------- UI ----------------
console.print(
    Panel.fit(
        "[bold green]LangGraph RAG Agent Ready[/bold green]\n"
        "[cyan]Logging + Tool tracing enabled[/cyan]\n"
        "[red]Type 'exit' to quit[/red]",
        border_style="green",
    )
)


# ---------------- Loop ----------------
state = {
    "messages": []
}

while True:
    question = console.input("\nQuestion > ")

    if question.lower() == "exit":
        break

    logger.info(f"USER QUERY: {question}")

    state["messages"].append(HumanMessage(content=question))

    final_answer = ""

    for event in app.stream(state, stream_mode="values"):

        messages = event["messages"]

        last_msg = messages[-1]

        # show tool traces in logs automatically
        if last_msg.type == "tool":
            logger.info(f"TOOL OUTPUT: {last_msg.content[:300]}...")

        if last_msg.type == "ai":
            final_answer = last_msg.content

    console.print(
        Panel(
            Markdown(final_answer.strip() if final_answer else "No response generated"),
            title="Output",
            border_style="green",
            expand=False
        )
    )
