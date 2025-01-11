from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from openai import OpenAI

from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import START, END
from langgraph.graph.message import add_messages
from langchain.tools.retriever import create_retriever_tool
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated
from typing_extensions import TypedDict

import qdrant_tool

import io, os
import time, json
from PIL import Image

QDRANT_HOST = os.getenv('QDRANT_HOST','localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT','6555'))
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION','Gruppo1')

class State(TypedDict):
    messages: Annotated[list, add_messages]

def show_graph(graph: StateGraph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()

        img = Image.open(io.BytesIO(png_bytes))
        img.save('simple-graph.jpg')

    except Exception as e:
        print("Could not generate or display the graph image in memory. Ensure all necessary dependencies are installed.")
        print("Error:", e)

llama = ChatOllama(model = "llama3.2:3b", temperature = 0.7, num_predict = 256, base_url="http://192.168.1.24:11434")
qwq = ChatOllama(model = "qwq", temperature = 0.8, num_predict = 1024, base_url="http://192.168.1.24:11434")
deep_seek = ChatOpenAI(model="deepseek-chat", api_key=os.getenv('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com", temperature=0.8, max_tokens=2048)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if DEEPSEEK_API_KEY:
    writer_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.8,
        max_tokens=2048
    )
else:
    ## num_ctx is llama.context_length for llama3.2:3b
    writer_llm = ChatOllama(model = "llama3.2:3b", temperature = 0.8, num_predict = 2048, num_ctx=131072,base_url="http://192.168.1.24:11434")


deep_seek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
)

def send_messages(messages, tools):
    new_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            if 'role' not in msg:
                msg['role'] = 'user'
            new_messages.append(msg)
        else:
            new_messages.append({'role': 'user', 'content': str(msg)})
    response = deep_seek_client.chat.completions.create(
        model="deepseek-chat",
        messages=new_messages,
        tools=tools
    )
    choice = response.choices[0]
    result = {
        "role": choice.message.role,
        "content": choice.message.content,
        "tool_calls": getattr(choice, "tool_calls", [])
    }
    return result




def generate_references(papers: list) -> str:
    """
    Returns a references paragraph based on the metadata of the given papers.
    Args:
        `papers` (`list(dict)`)
    
    Returns:
        `str`
    """
    references = []
    for i,paper in enumerate(papers,1):
        authors = paper['authors'].split(', ')
        first_author = authors[0].split(' ')[-1] + ', ' + ' '.join(authors[0].split(' ')[:-1])
        year = paper['published'].split('-')[0]
        citation = f'({i}) {first_author}{" et al." if len(authors) > 1 else ""}'
        citation += f' "' + paper['title'] + f'." ({year}), ' + paper['link']
        references.append(citation)
    return '\n\n'.join(references)

qdrant_retriever = create_retriever_tool(
    qdrant_tool.Qdrant_tool(host=QDRANT_HOST,port=QDRANT_PORT,collection=QDRANT_COLLECTION,top_k=2),
    "retrieve_arxiv_papers",
    "Search and return arxiv papers that are related to the requested query.",
)

def researcher(state: State):
    print("In researcher")
    messages = state["messages"]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_arxiv_papers",
                "description": "Search and return arxiv papers that are related to the requested query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query for arxiv papers.",
                        }
                    },
                    "required": ["query"]
                },
            }
        },
    ]
    message = send_messages(messages, tools)
    if not message.get("tool_calls"):
        messages.append(message)
        return {"messages": [message]}
    
    
    messages.append(message)
    messages.append({
        "role": "assistant",
        "content": "Results from retriever tool"
    })
    message = send_messages(messages, tools)
    return {"messages": [message]}

writer_prompt = PromptTemplate(
    template="""You are an agent tasked with writing a formal literature review article (also known as a survey) based on the user's query and the provided academic papers.\n
    Here are the papers: \n\n {context} \n\n
    Here is the user query: {question} \n
    The survey must be formal and should have an overview of all the papers and a short breakdown of every paper given.
    You must only focus on the given papers.\n
    Start directly with the content of the review. Avoid phrases like 'Based on the provided papers' or 'I will create a survey'.
    Do not write references at the end.
    """,
    input_variables=["context", "question"],
)


def writer(state: State):
    print('In writer')
    messages = state["messages"]

    question = messages[0].content
    docs = messages[-1].content

    metadata = []
    for doc in docs.split('\n\n'):
        if doc.strip():
            try:
                metadata.append(json.loads(doc))
            except json.JSONDecodeError:
                pass

    model = writer_prompt | writer_llm
    response = model.invoke({'context':docs, 'question': question})

    references = generate_references(metadata)

    with open(f'temp/output-{str(int(time.time()))}.txt','w') as f:
        f.write(response.content)

    response.content += '\n\n###REFERENCES\n\n' + references

    return {"messages": [response]}


def get_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("researcher",researcher)
    graph_builder.add_node("writer",writer)
    retrieve = ToolNode([qdrant_retriever])
    graph_builder.add_node("retriever",retrieve)

    graph_builder.add_edge(START, "researcher")
    graph_builder.add_edge("researcher", "retriever")
    graph_builder.add_edge("retriever", "writer")
    graph_builder.add_edge("writer", END)

    memory = MemorySaver()

    graph = graph_builder.compile(checkpointer=memory)

    return graph