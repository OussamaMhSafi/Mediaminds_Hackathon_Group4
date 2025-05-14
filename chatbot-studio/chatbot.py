from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel

# We will use this model for the conversation
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)  # Using GPT-4o which has vision capabilities

# State class to store messages and image data
class State(MessagesState):
    image_path: Optional[str] = None
    image_data: Optional[Dict[str, Any]] = None
    # Add a wait_for_input flag to control the conversation flow
    wait_for_input: bool = False

# Function to load and process the image
def load_image(state: State) -> Dict[str, Any]:
    if not state.get("image_path"):
        return state
    
    try:
        image = Image.open(state["image_path"])
        buffered = BytesIO()
        image.save(buffered, format=image.format or "JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "image_data": {
                "success": True,
                "image_data": img_str,
                "width": image.width,
                "height": image.height,
                "format": image.format
            }
        }
    except Exception as e:
        return {
            "messages": [
                HumanMessage(content=f"Error processing image: {str(e)}")
            ]
        }

# Define the logic to call the model
def call_model(state: State):
    messages = state["messages"]
    image_data = state.get("image_data")
    
    # Check if there's image data and if the last message doesn't contain an image
    if image_data and image_data.get("success"):
        # Get the last message content (if any)
        last_message_content = ""
        has_image_in_message = False
        
        if messages and isinstance(messages[-1].content, list):
            # Check if the last message already has an image
            for content_item in messages[-1].content:
                if isinstance(content_item, dict) and content_item.get("type") == "image_url":
                    has_image_in_message = True
                    break
        
        if messages and not has_image_in_message:
            if isinstance(messages[-1].content, str):
                last_message_content = messages[-1].content
            # Remove the last text-only message as we'll replace it with a multimodal message
            messages = messages[:-1]
        
        # Create a new message with both text and image
        multimodal_content = []
        
        # Add text if it exists
        if last_message_content:
            multimodal_content.append({
                "type": "text", 
                "text": last_message_content
            })
        else:
            multimodal_content.append({
                "type": "text", 
                "text": "Please describe this image and be as specific as possible. If the image contained people find out if they resemble any well-known figure in our history:"
            })
            
        # Add image
        multimodal_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_data['image_data']}"
            }
        })
        
        # Add the multimodal message
        messages.append(HumanMessage(content=multimodal_content))
    
    # Generate response
    response = model.invoke(messages)
    
    # Set wait_for_input to True to indicate we need user input before continuing
    return {"messages": response, "wait_for_input": True}

# Decision function to determine the next step in the conversation
def decide_next_step(state: State) -> Literal["wait_for_input", "continue_conversation"]:
    # If wait_for_input is True, we need user input before continuing
    if state.get("wait_for_input", False):
        return "wait_for_input"
    else:
        return "continue_conversation"

# Function to handle user input and prepare for the next turn
def handle_user_input(state: State) -> Dict[str, Any]:
    # Reset the wait_for_input flag and image_data for the next turn
    return {
        "wait_for_input": False,
        "image_data": None,
        "image_path": None
    }

# Create the workflow graph
def create_chatbot_graph():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("load_image", load_image)
    workflow.add_node("conversation", call_model)
    workflow.add_node("handle_user_input", handle_user_input)
    
    # Set the connections
    workflow.add_edge(START, "load_image")
    workflow.add_edge("load_image", "conversation")
    
    # Add conditional edges from conversation
    workflow.add_conditional_edges(
        "conversation",
        decide_next_step,
        {
            "wait_for_input": END,  # If waiting for input, end this execution
            "continue_conversation": "handle_user_input"  # Otherwise, process and continue
        }
    )
    
    workflow.add_edge("handle_user_input", "load_image")  # Complete the loop
    
    # Compile the graph
    return workflow.compile()

# Create the graph
graph = create_chatbot_graph()
