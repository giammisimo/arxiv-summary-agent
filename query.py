import SummaryAgent
from langchain_core.messages.tool import ToolMessage
from random import randint

graph = SummaryAgent.get_graph()
SummaryAgent.show_graph(graph)

def main():
    config = {"configurable": {"thread_id": str(randint(1,2048))}, "use_memory": False}

    user_input = input("Query: ")

    events = graph.stream(
        {"messages": [{"role":"user","content": user_input}]}, config, stream_mode="values"
    )
    for event in events:
        #print(event)
        #input('Press Enter to continue')
        last_message = event["messages"][-1]
        if not isinstance(last_message,ToolMessage):
            last_message.pretty_print() #stream_mode='value'
        else:
            print('ToolMessage\n')
    print(last_message.response_metadata)
    print(last_message.usage_metadata)

if __name__ == "__main__":
    main()
