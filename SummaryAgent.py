from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
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
MAX_MESSAGES = 10

class State(TypedDict):
    messages: Annotated[list, add_messages]

def show_graph(graph: StateGraph):
    try:
        png_bytes = graph.get_graph().draw_mermaid_png()

        img = Image.open(io.BytesIO(png_bytes))
        img.save('summaryagent.jpg')

    except Exception as e:
        print("Could not generate or display the graph image in memory. Ensure all necessary dependencies are installed.")
        print("Error:", e)

#llama = ChatOllama(model = "llama3.2:3b", temperature = 0.7, num_predict = 256, base_url="http://192.168.1.24:11434")
#qwq = ChatOllama(model = "qwq", temperature = 0.8, num_predict = 1024, base_url="http://192.168.1.24:11434")

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
if DEEPSEEK_API_KEY:
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.8,
        max_tokens=4096
    )
else:
    ## num_ctx is llama.context_length for llama3.2:3b
    llm = ChatOllama(model = "llama3.2:3b", temperature = 0.8, num_predict = 2048, num_ctx=131072,base_url="http://192.168.1.24:11434")


def has_papers(state: State):
    """
    Returns True if papers were found in the retrieval, False otherwise
    """
    messages = state["messages"]

    if len(messages) >= MAX_MESSAGES:
        print(f"Max retries ({MAX_MESSAGES}) reached. Ending chain.")
        return True

    has_content = len(messages[-1].content) > 0
    return has_content


## This is all a mess of str concatenations - Thanks python 3.9!
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
    qdrant_tool.Qdrant_tool(host=QDRANT_HOST,port=QDRANT_PORT,collection=QDRANT_COLLECTION,top_k=5),
    "retrieve_arxiv_papers",
    "Search and return arxiv papers that are related to the requested query.",
)

def researcher(state: State):
    print("In researcher")
    messages = state["messages"]
    model = llm
    model = model.bind_tools([qdrant_retriever])
    response = model.invoke(messages)
    return {"messages": [response]}


evaluator_prompt = PromptTemplate(
    template="""
    Rewrite the following query to extract only the core keywords directly related to the main topic.\n
    Keep the query concise and focused, avoiding unnecessary expansions or additional terms.\n
    Expand any acronyms or ambiguous terms where necessary.\n
    Avoid adding any explanations or introductory phrases. Respond only with the rewritten keywords and nothing else.\n

    Original query: '{original_query}'
    """,
    input_variables=["original_query"],
)


def evaluator(state: State):
    print("In evaluator")
    messages = state["messages"]

    original_query = messages[0].content

    model = evaluator_prompt | llm
    response = model.invoke({"original_query": original_query})

    print(f"Rewritten query: {response.content.strip()}")

    messages = response

    return {"messages": messages}


writer_prompt = PromptTemplate(
    template="""You are an agent tasked with writing a formal literature review article (also known as a survey) based on the user's query and the provided academic papers.\n
    Here are the papers: \n\n {context} \n\n
    Here is the user query: {question} \n
    The survey must be formal and should have: an single summary that talks about the argument and the papers and a short breakdown of every paper given.
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

    if len(docs) == 0:
        return {
            "messages": [
                AIMessage(content=f"I apologize, but after {MAX_MESSAGES} attempts to refine the search query, "
                         "I still couldn't find any relevant papers. You might want to try a different search term or broaden your query.")
            ]
        }

    metadata = [json.loads(doc) for doc in (docs.split('\n\n'))]

    tokens = 0
    for i,paper in enumerate(metadata):
        tokens += len(paper['text'])
        print(tokens/3)
        if (tokens/3) > 52000:
            print(f'TOO MANY TOKENS, LIMITED TO {i} PAPERS')
            docs = '\n\n'.join([json.dumps(metadata[k]) for k in range(i)])
            references = generate_references(metadata[:i])
            break
    else:
        references = generate_references(metadata)

    print('Estimated tokens:',tokens/3)

    model = writer_prompt | llm
    response = model.invoke({'context':docs, 'question': question})

    with open(f'temp/output-{str(int(time.time()))}.txt','w') as f:
        f.write(response.content)

    response.content += '\n\n---\n### REFERENCES\n\n' + references

    return {"messages": [response]}


def get_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("researcher",researcher)
    graph_builder.add_node("evaluator", evaluator)
    graph_builder.add_node("writer",writer)
    retrieve = ToolNode([qdrant_retriever])
    graph_builder.add_node("retriever",retrieve)

    graph_builder.add_edge(START, "researcher")
    graph_builder.add_edge("researcher", "retriever")
    graph_builder.add_conditional_edges(
        "retriever",
        has_papers,
        {
            True: "writer",
            False: "evaluator"
        }
    )
    graph_builder.add_edge("evaluator", "researcher")
    graph_builder.add_edge("writer", END)

    memory = MemorySaver()

    graph = graph_builder.compile(checkpointer=memory)

    return graph
