"""
This module serves to integrate the LLM with the tools and initiate the main application
1. Make the state class
-- prepare the system prompt important
2. Create the nodes and the LLM
3. Create the stategraph

"""

from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.prebuilt import ToolNode  # imp shortcut to make the tool node
from langgraph.graph import StateGraph, START, END
from retriever import CustomRetriever  # custom import I made be cautious
from tools import CustomRetrieverTool, web_search_tool  # custom tools made
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialization code
## Initialize retriever (do this once at startup)
print("[INIT] Initializing retriever...")
retriever_instance = CustomRetriever("sample_data.csv")  # sample file for testing
vector_retriever = retriever_instance.run()
print("[INIT] Retriever ready.")
retriever_tool = CustomRetrieverTool(retriever=vector_retriever)
## Initialize the parser
parser = JsonOutputParser()
print("[INIT] Output parser ready.")

# LLM
llm_instance = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
tools = [retriever_tool, web_search_tool]
llm_instance = llm_instance.bind_tools(tools)  # tool binded llm ready
print("[INIT] Tool binded LLM ready.")

# State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Nodes
def llm(state: AgentState) -> AgentState:
    """Node to send the query to the llm"""
    sys_prompt = """
    You are a friendly assistant for Sumit or Sonu or Steo. Always use his name whenever possible. Your core behaviors include these adjectives:
    motherly, helpful, optimistic.
    You have access to these following tools:
    1. retriever_tool: to retrieve the most relevant information about Sumit's notes.
    2. web_search_tool: to search the internet for more information.
    Strictly use the following WORKFLOW CYCLE to think:
    Question: <the input query>
    Thought: always try to find the best context about the user query, ask follow up questions if needed, decide if I need to use tools
    Action: user tools if needed
    Observation: analyze the output of the tools 
    Thought: based on the observation, decide if you need to repeat or give the final answer
    Action: Give final answer or repeat the Thought, Action, Observation loop
    
    How to give the final answer?
    Strictly use the JSON format:
    {{
    "Final Answer": "<greetings> <final answer to the query>"
    "confidence": "<high/medium/low>"
    }}
    Example use case to give final answer:
    {{
        "Final Answer": "Hello Steo! According to your notes, you were doing 'Gym session' at 6 PM.",
        "confidence": "high"
    }}
    
    Now begin! Remember to follow the WORKFLOW CYCLE and use the JSON format for the final answer.
    """

    message = state["messages"]
    response = llm_instance.invoke([SystemMessage(content=sys_prompt)] + list(message))
    # Poor performance code need improvement later
    # parse json if final answer
    # Only parse if it's the final answer (no tool calls)
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        try:
            parsed = parser.parse(response.content)
            if "Final Answer" in parsed:
                return {"messages": [AIMessage(content=str(parsed))]}
        except Exception:
            pass  # Not JSON yet, continue workflow
    return {"messages": [response]}

def should_continue(state: AgentState) -> bool:
    """Node to decide if to continue to call tools or end the process and give the final answer"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return True
    return False

# StateGraph

graph = StateGraph(AgentState)
graph.add_node("llm_node", llm)

# graph.add_node("should_continue_node", should_continue) (not needed check)
tool_node = ToolNode(tools=tools)
graph.add_node("tool_node", tool_node)  # tool node using prebuild ToolNode code
# Edges
graph.add_edge(START, "llm_node")
graph.add_conditional_edges(
    "llm_node",
    should_continue,
    {
        True: "tool_node",
        False: END
    }
)
graph.add_edge("tool_node", "llm_node")  # reconnection
app = graph.compile()

# test
response = app.invoke({"messages": [HumanMessage(content="At what time I was doing 'Gym session' according to my notes ")]})
print("\n\nFinal Response from Agent:\n", response["messages"][-1].content[0]['text'])