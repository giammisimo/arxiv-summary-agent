
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool

from typing import Annotated
from typing_extensions import TypedDict

import qdrant_tool

import io, os
from PIL import Image

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
llama_long = ChatOllama(model = "llama3.2:3b", temperature = 0.8, num_predict = 1024, base_url="http://192.168.1.24:11434")
qwq = ChatOllama(model = "qwq", temperature = 0.8, num_predict = 1024, base_url="http://192.168.1.24:11434")

'''DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com",
    temperature=0.8,
    max_tokens=256
)'''

qdrant_retriever = create_retriever_tool(
    qdrant_tool.Qdrant_tool(host='192.168.1.26',port=6555,collection='Gruppo1',top_k=2),
    "retrieve_arxiv_papers",
    "Search and return arxiv papers that are related to the requested query.",
)

def researcher(state: State):
    print("In researcher")
    messages = state["messages"]
    model = llama
    model = model.bind_tools([qdrant_retriever])
    response = model.invoke(messages)
    return {"messages": [response]}


# Write the survey inside <survey></survey> tags.

writer_prompt = PromptTemplate(
    template="""You are an agent tasked with writing a survey based on the query given by the user and the papers that you are given.\n 
    Here are the papers: \n\n {context} \n\n
    Here is the user query: {question} \n
    The survey must be formal and should have an overview of all the papers and a short breakdown of every paper given.
    You must only focus on the given papers.
    """,
    input_variables=["context", "question"],
)

def writer(state: State):
    print('In writer')
    print(state)
    messages = state["messages"]

    question = messages[0].content
    docs = messages[-1].content

    model = writer_prompt | llama_long
    response = model.invoke({'context':docs, 'question': question})

    print(response)
    return {"messages": [response]}

def evaluator(state: State):
    print('In evaluator')
    #print(state)
    exit()
    return {'messages':state['messages']}

def should_continue(state: State):
    return END # placeholder

memory = MemorySaver()

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("researcher",researcher)
graph_builder.add_node("writer",writer)
graph_builder.add_node("evaluator",evaluator)
retrieve = ToolNode([qdrant_retriever])
graph_builder.add_node("retriever",retrieve)

'''graph_builder.add_conditional_edges(
    "researcher",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retriever",
        END: "writer",
    },
)'''

graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "retriever")
graph_builder.add_edge("retriever", "writer")
graph_builder.add_edge("writer", "evaluator")
graph_builder.add_conditional_edges("evaluator", should_continue, ["writer",END])

graph = graph_builder.compile()

show_graph(graph)
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("Query: ")
    #user_input = 'llm-based agents'
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
# The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role":"user","content": user_input}]}, config, stream_mode="debug"
    )
    for event in events:
        #print(event)
        input('Press Enter to continue')
        #event["messages"][-1].pretty_print() #stream_mode='value'
        #stream_mode='debug'
        '''if event['type'] == 'checkpoint':
            if event['payload']['values']["messages"]:
                event['payload']['values']["messages"][-1].pretty_print() 
        elif event['type'] == 'task':
            event['payload']['input']["messages"][-1].pretty_print() 
        elif event['type'] == 'value':
            event["messages"][-1].pretty_print()
        elif event['type'] == 'task_result':
            print('task')
            event["payload"]['result'][0][-1][0].pretty_print()
        else:
            event["messages"][-1].pretty_print()'''
        print(event['type'])

