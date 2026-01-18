import langgraph #v0.0.38
from langgraph.graph.message import MessageGraph


# Define simple node functions
def greet(messages):
    return ["Hello!"]

def ask_name(messages):
    return ["What is your name?"]

def end_conversation(messages):
    return ["Goodbye!"]

# Create the MessageGraph
workflow = MessageGraph()

# Add nodes
workflow.add_node("greet", greet)
workflow.add_node("ask_name", ask_name)
workflow.add_node("end", end_conversation)

# Set entry point
workflow.set_entry_point("greet")
workflow.set_finish_point("end")   # ⭐ 關鍵

# Add edges
workflow.add_edge("greet", "ask_name")
workflow.add_edge("ask_name", "end")

# Compile the workflow
runnable = workflow.compile()

# Run the workflow with an empty message list
result = runnable.invoke([])
print(result)