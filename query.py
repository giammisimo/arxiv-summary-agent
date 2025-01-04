import simple_system
from langchain_core.messages.tool import ToolMessage

graph = simple_system.get_graph()
simple_system.show_graph(graph)

def main():
    config = {"configurable": {"thread_id": "1"}}


    user_input = input("Query: ")
    #user_input = 'llm-based agents'

    # The config is the **second positional argument** to stream() or invoke()!
    events = graph.stream(
        {"messages": [{"role":"user","content": user_input}]}, config, stream_mode="values"
    )
    for event in events:
        #print(event)
        input('Press Enter to continue')
        last_message = event["messages"][-1]
        if not isinstance(last_message,ToolMessage):
            last_message.pretty_print() #stream_mode='value'
        else:
            print('ToolMessage\n')
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
        #print(event['type'])

if __name__ == "__main__":
    main()