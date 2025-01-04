from fastapi import FastAPI
from pydantic import BaseModel

from langchain_core.messages.tool import ToolMessage
import simple_system

app = FastAPI()
graph = simple_system.get_graph()

class UserMessage(BaseModel):
    query: str

@app.get("/")
async def check():
    return ({"status": "up"}), 200

@app.post("/query")
async def check(message: UserMessage):
    config = {"configurable": {"thread_id": "1"}}
    user_input = message.query
    
    events = graph.stream(
        {"messages": [{"role":"user","content": user_input}]}, config, stream_mode="values"
    )

    last_message = None
    for event in events:
        last_message = event["messages"][-1]
        if not isinstance(last_message,ToolMessage):
            last_message.pretty_print() #stream_mode='value'
        else:
            print('ToolMessage\n')
    
    return {'message':last_message.content}