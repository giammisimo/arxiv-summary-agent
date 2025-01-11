import simple_system
from langchain_core.messages.tool import ToolMessage

graph = simple_system.get_graph()
simple_system.show_graph(graph)

def main():
    config = {"configurable": {"thread_id": "1"}}

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