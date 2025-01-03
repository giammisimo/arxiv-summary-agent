from typing import Annotated, Literal

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.types import Command
from langgraph.prebuilt import ToolNode, tools_condition, create_react_agent
from langgraph.graph import StateGraph, MessagesState
from langgraph.graph.graph import START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain.agents.agent import AgentExecutor
from typing_extensions import TypedDict

import arxiv_tool, qdrant_tool

import io, os
import re
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


def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


llm = ChatOllama(model = "llama3.2:3b", temperature = 0.7, num_predict = 256, base_url="http://192.168.1.24:11434")

'''DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

llm = ChatOpenAI(
    model="deepseek-chat", 
    api_key=DEEPSEEK_API_KEY, 
    base_url="https://api.deepseek.com",
    temperature=0.8,
    max_tokens=256
)'''


qdrant_retriever = create_retriever_tool(
    qdrant_tool.Qdrant_tool(host='192.168.1.26',port=6555,collection='Gruppo1'),
    "retrieve_arxiv_papers",
    "Search and return arxiv papers that are related to the requested query.",
)

researcher_prompt = '''You are an agents tasked with searching the most related papers about the topic requested by the user.
You are given a tool called retrieve_arxiv_papers that will return a list of papers related to the query.'''

researcher_prompt = '''Call retrieve_arxiv_papers with the query of the user.'''

researcher_agent = create_react_agent(llm, tools=[qdrant_retriever], state_modifier=researcher_prompt)
researcher = AgentExecutor.from_agent_and_tools(researcher_agent, [qdrant_retriever])

#def researcher(state: State):
#    print('In researcher')
#    print(state)
#    input('>')
#    return {"messages": [
#                HumanMessage(content=researcher_agent.invoke(state)['messages'][-1].content, name="researcher")
#            ]}

def writer(state: State):
    print('In writer')
    print(state)
    input('>')
    exit()
    return researcher(state)

def evaluator(state: State):
    print('In evaluator')
    print(state)
    input('>')
    exit()
    return researcher(state)

def should_continue(state: State):
    messages = state["messages"]
    #print(messages)
    last_message = messages[-1]
    if 'researcher' in last_message.content.lower():
        print('Calling researcher....')
        return "researcher"
    elif 'developer' in last_message.content.lower():
        print('Calling developer....')
        return "developer"
    else:
        print('Ending....')
        return END

memory = MemorySaver()

graph_builder = StateGraph(MessagesState)
graph_builder.add_node("researcher",researcher)
graph_builder.add_node("writer",writer)
graph_builder.add_node("evaluator",evaluator)

graph_builder.add_edge(START, "researcher")
graph_builder.add_edge("researcher", "writer")
graph_builder.add_edge("writer", "evaluator")
graph_builder.add_conditional_edges("evaluator", should_continue, ["writer",END])

graph = graph_builder.compile(checkpointer=memory)

show_graph(graph)
config = {"configurable": {"thread_id": "1"}}

while True:
    #user_input = input("Query: ")
    user_input = 'llm-based agents'
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
# The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role":"user","content": user_input}]}, config, stream_mode="debug"
    )
    for event in events:
        print(event)
        #event["messages"][-1].pretty_print() #stream_mode='value'
        #stream_mode='debug'
        if event['type'] == 'checkpoint':
            if event['payload']['values']["messages"]:
                event['payload']['values']["messages"][-1].pretty_print() 
        elif event['type'] == 'task':
            event['payload']['input']["messages"][-1].pretty_print() 
        elif event['type'] == 'value':
            event["messages"][-1].pretty_print()
        else:
            event["messages"][-1].pretty_print()

