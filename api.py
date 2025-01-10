from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from langchain_core.messages.tool import ToolMessage
import os
import simple_system
import qdrant_tool

QDRANT_HOST = os.getenv('QDRANT_HOST','localhost')
QDRANT_PORT = int(os.getenv('QDRANT_PORT','6555'))
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION','Gruppo1')

app = FastAPI()
graph = simple_system.get_graph()
qdrant = qdrant_tool.Qdrant_tool(host=QDRANT_HOST,port=QDRANT_PORT,collection=QDRANT_COLLECTION,top_k=10)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class UserMessage(BaseModel):
    query: str

class AgentResponse(BaseModel):
    message: str

class QueryResult(BaseModel):
    arxiv_id: str
    id_uuid: str 
    published: str
    score: float 
    summary: str 
    title: str 

@app.get("/")
async def check():
    return ({"status": "up"}), 200

@app.post("/agent", response_model=AgentResponse)
async def agent(message: UserMessage):
    config = {"configurable": {"thread_id": "1"}}
    user_input = message.query
    
    events = graph.stream(
        {"messages": [{"role":"user","content": user_input}]}, config, stream_mode="values"
    )

    print("API key", os.getenv('DEEPSEEK_API_KEY'))
    print('QUERY:', user_input)
    last_message = None
    for event in events:
        last_message = event["messages"][-1]
        #if not isinstance(last_message,ToolMessage):
        #    last_message.pretty_print() #stream_mode='value'
        #else:
        #    print('ToolMessage\n')
    print(last_message.response_metadata)
    print(last_message.usage_metadata)

    return {'message':last_message.content}

@app.post("/query", response_model=List[QueryResult])
async def query(message: UserMessage):
    if not message:
        return HTTPException(status_code=400, detail={"error": "Il campo 'query' è obbligatorio"})

    try:
        results = qdrant.embed_and_search(QDRANT_COLLECTION,message.query,10)

        if results:
            response = []
            for result in results:
                response.append(QueryResult(
                    id_uuid = result.id,
                    score = float(result.score),
                    title = result.payload.get('title', 'N/A'),
                    summary = result.payload.get('summary', 'N/A'),
                    arxiv_id = result.payload.get('arxiv-id', 'N/A'),
                    published = result.payload.get('published', 'N/A')
                    )
                )
            return response
        else:
            return HTTPException(status_code=404, detail={"message": "Nessun risultato trovato."})
    except Exception as e:
        return HTTPException(status_code=500, detail={"error": str(e)})