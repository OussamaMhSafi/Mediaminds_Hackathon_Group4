from typing import TypedDict, List, Dict, Any, Optional, Annotated, Literal
from pydantic import BaseModel, Field
from enum import Enum
import operator

from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableConfig

from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode

# Define the state which will be passed around
class VerificationState(TypedDict):

    decision: Optional[Dict[str, Any]]

def create_verification_graph():
    workflow = StateGraph(VerificationState)
    
    # Add nodes like this, nodes can represent a function
    workflow.add_node("initialize", lambda state: state)
    
    # Compile the graph
    return workflow.compile()

# Create the graph, to launche using 'langgraph dev'
graph = create_verification_graph()
