# # main expression agent 
# # drug selection agent
# # drug drug interaction agent 
# # drug improvement 

# from MainExpressionAgent import MainExpressionAgent
# from agents import Runner
# import asyncio

# async def main():
#     open("synthesis_report.md", "w").close()
#     # Clear out checkpoints folder
#     import shutil
#     shutil.rmtree("checkpoints", ignore_errors=True)
#     result = await Runner.run(MainExpressionAgent, f"prompt: Find a combination of approved or post-phase-1 clinical trial drugs that can treat Alzheimer's.", max_turns=300)
#     print(result.final_output)


# if __name__ == "__main__":
#     asyncio.run(main())


import streamlit as st
import asyncio
import shutil
import os

# --- Agent Imports ---
from MainExpressionAgent import MainExpressionAgent 
from agents import Runner
from agents.items import ToolCallItem
from openai.types.responses import ResponseTextDeltaEvent, ResponseFunctionToolCall
from agents.stream_events import RunItemStreamEvent 

# --- Page Configuration ---
st.set_page_config(
    page_title="Drug Repurposing Agent",
    page_icon="💊",
    layout="wide"  # Changed to wide layout
)

# Custom CSS for handoff styling
st.markdown("""
<style>
    .handoff-message {
        background-color: #f0f7ff;
        border: 2px solid #4a90e2;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .handoff-indicator {
        color: #4a90e2;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .handoff-arrow {
        color: #4a90e2;
        font-size: 20px;
        text-align: center;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Backend Logic with Final Streaming Fix ---
async def run_agent_workflow(prompt: str):
    """
    Runs the agent workflow and streams events back to the Streamlit UI.
    """
    # Clean up previous run data
    open("synthesis_report.md", "w").close()
    shutil.rmtree("checkpoints", ignore_errors=True)
    os.mkdir("checkpoints")
    
    with st.status("🔬 Agent workflow initiated...", expanded=True) as status:
        answer_placeholder = st.empty()
        full_response = ""
        current_agent = "MainExpressionAgent"

        try:
            result = Runner.run_streamed(
                MainExpressionAgent, 
                f"prompt: {prompt}", 
                max_turns=300
            )

            async for event in result.stream_events():
                # Event 1: LLM token output for the final answer
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    full_response += event.data.delta
                    answer_placeholder.markdown(full_response + "▌")

                # Event 2: Agent action/tool call updates
                elif event.type == "run_item_stream_event":
                    if isinstance(event.item, ToolCallItem):
                        raw_tool_call = event.item.raw_item
                        
                        if isinstance(raw_tool_call, ResponseFunctionToolCall):
                            tool_name = raw_tool_call.name
                            
                            # Check if this is a handoff
                            if tool_name.startswith("transfer_to_"):
                                new_agent = tool_name.replace("transfer_to_", "").replace("_", " ")
                                # Add handoff visual
                                st.markdown(f"""
                                    <div class="handoff-message">
                                        <div class="handoff-indicator">🔄 Agent Handoff</div>
                                        <div>From: {current_agent}</div>
                                        <div class="handoff-arrow">↓</div>
                                        <div>To: {new_agent}</div>
                                    </div>
                                """, unsafe_allow_html=True)
                                current_agent = new_agent
                            else:
                                status.update(label=f"🤖 Using tool: `{tool_name}`...")

            answer_placeholder.markdown(full_response)
            status.update(label="✅ Workflow Complete!", state="complete")

        except Exception as e:
            st.error(f"An error occurred during the agent run: {e}")
            status.update(label="❗️ Error", state="error")
            full_response = f"Error: {e}"

    return full_response


# --- 2. Simplified Single-Column Streamlit UI ---
st.title("💊 Drug Repurposing Agent Workflow")
st.caption("An interactive UI for finding novel drug combinations using OpenAI Agents.")

# Container for the chat interface
chat_container = st.container(height=600, border=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "What combination of drugs would you like me to research today?"}
    ]

# Display prior messages
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Main chat input
if prompt := st.chat_input("e.g., Treat Alzheimer's Disease"):
    # Add and display the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

    # Display the agent's response stream
    with chat_container:
        with st.chat_message("assistant"):
            final_result = asyncio.run(run_agent_workflow(prompt))
    
    # Add the final result to history
    st.session_state.messages.append({"role": "assistant", "content": final_result})
