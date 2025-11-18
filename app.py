import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
import importlib
import json
import threading


# Page configuration
st.set_page_config(
    page_title="IYP Assistant Agent",
    page_icon="🌐",
    layout="wide"
)

APP_TITLE = "Internet Yellow Pages Assistant Agent"
APP_TAGLINE = "Ask questions about global Internet infrastructure and get answers straight from IYP."
WARNING_TEXT = (
    "Always validate generated Cypher first: expand the queries under each answer to verify or adjust them."
)
SAMPLE_QUESTIONS = [
    "Find prefixes originated by AS 2497.",
    "Which IXPs are located in Greece?",
    "List the ASNs registered as Vodafone (AS3329) customers.",
    "Surface facilities located in Heraklion.",
]
IYP_CONSOLE_URL = "http://iyp.iijlab.net/"

st.title(APP_TITLE)
st.markdown(APP_TAGLINE)


st.markdown(
    """
    <style>
    button[kind="primary"] {
        background-color: #d9534f !important;
        border-color: #d9534f !important;
        color: #ffffff !important;
    }
    button[kind="primary"]:hover {
        background-color: #c9302c !important;
        border-color: #c12e2a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to extract response data
def extract_response(response):
    messages = response.get("messages", [])
    
    # Extract user question
    human_msg = next(
        (m for m in messages if getattr(m, "role", None) == "user" or m.__class__.__name__ == "HumanMessage"),
        None,
    )
    user_question = ""
    if human_msg:
        content = getattr(human_msg, 'content', '')
        user_question = content
    
    ai_msgs = [m for m in messages if getattr(m, "role", None) == "assistant" or m.__class__.__name__ == "AIMessage"]
    ai_answer = getattr(ai_msgs[-1], "content", "") if ai_msgs else ""
    
    # Extract all Cypher queries and their results
    cypher_queries = []
    cypher_results = []
    
    for i, msg in enumerate(messages):
        msg_type = msg.__class__.__name__
        
        # Capture Cypher queries from AI tool calls
        if msg_type == "AIMessage":
            tool_calls = getattr(msg, 'tool_calls', [])
            for tool_call in tool_calls:
                tool_name = tool_call.get('name', '')
                if 'cypher' in tool_name.lower() and tool_name != 'get_neo4j_schema':
                    args = tool_call.get('args', {})
                    cypher_query = args.get('cypher', args.get('query', args.get('statement', '')))
                    if cypher_query:
                        cypher_queries.append({
                            'query': cypher_query,
                            'tool_call_id': tool_call.get('id', ''),
                            'tool_name': tool_name,
                            'all_args': args
                        })
        
        # Capture tool results
        elif msg_type == "ToolMessage":
            content = getattr(msg, 'content', '')
            tool_call_id = getattr(msg, 'tool_call_id', '')
            tool_name = getattr(msg, 'name', '')
            
            result_data = None
            try:
                result_data = json.loads(content)
            except json.JSONDecodeError:
                result_data = content
            
            cypher_results.append({
                'tool_call_id': tool_call_id,
                'tool_name': tool_name,
                'result': result_data,
                'raw_content': content
            })
    
    return {
        "question": user_question,
        "llm_answer": ai_answer,
        "cypher_queries": cypher_queries,
        "cypher_results": cypher_results,
        "message_count": len(messages),
        "num_queries": len(cypher_queries),
        "num_results": len(cypher_results)
    }

def init_session_state():
    defaults = {
        "chat_history": [],
        "agent": None,
        "system_prompt": None,
        "memory_enabled": False,
        "memory_depth": 0,
        "user_api_key": "",
        "agent_key": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_session_state()


def ensure_background_loop():
    loop = st.session_state.get('_asyncio_loop')
    loop_thread = st.session_state.get('_asyncio_loop_thread')

    if loop is None or loop.is_closed() or loop_thread is None or not loop_thread.is_alive():
        loop = asyncio.new_event_loop()
        loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
        
        loop_thread.start()
        st.session_state._asyncio_loop = loop
        st.session_state._asyncio_loop_thread = loop_thread

    return loop


def run_async_task(coro):
    loop = ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def render_user_guidance():
    st.warning(WARNING_TEXT)
    st.markdown(
        "**Try questions like:**\n" + "\n".join(f"- {question}" for question in SAMPLE_QUESTIONS)
    )
    st.caption(f"IYP Console reference: {IYP_CONSOLE_URL}")


def build_conversation_messages(system_prompt, user_query):
    messages = [{"role": "system", "content": system_prompt}]

    history_depth = st.session_state.memory_depth if st.session_state.memory_enabled else 0
    history_depth = min(history_depth, 5)

    if history_depth and st.session_state.chat_history:
        recent_history = st.session_state.chat_history[-history_depth:]
        for chat in recent_history:
            past_user_content = f"Previous question: {chat['question']}"
            cypher_results = chat.get('cypher_results')
            if cypher_results:
                past_user_content += "\nCypher results:\n" + json.dumps(cypher_results, ensure_ascii=False, indent=2)

            messages.append({"role": "user", "content": past_user_content})
            messages.append({"role": "assistant", "content": chat.get('answer', '')})

    messages.append({"role": "user", "content": user_query})
    return messages

# Load system prompt and initialize agent
@st.cache_resource
def initialize_agent(api_key: str):
    # Load system prompt
    try:
        with open("system-prompt", "r", encoding="utf-8") as f:
            schema_and_examples = f.read()
    except FileNotFoundError:
        schema_and_examples = ""
        st.warning("Warning: system-prompt file not found.")
    
    system_prompt = f"""You are a helpful Neo4j Cypher query assistant with access to Neo4j database tools.

CRITICAL INSTRUCTIONS - YOU MUST FOLLOW THESE:
-You MUST use the read_neo4j_cypher tool to execute queries for EVERY question
-ALWAYS execute at least one Cypher query to get real data from the database
-Generate Cypher queries based ONLY on the provided schema below
-If a query returns insufficient data, refine it and execute another query
-Base your final answer ONLY on the actual query results you receive

{schema_and_examples}

REMEMBER: Never answer a question without first executing a Cypher query using the read_neo4j_cypher tool."""
    
    # Initialize MCP client synchronously
    async def get_tools():
        client = MultiServerMCPClient({
            "neo4j": {
                "command": "uvx",
                "args": ["mcp-neo4j-cypher@0.3.0", "--transport", "stdio"],
                "transport": "stdio",
                "env": {
                    "NEO4J_URI": "neo4j://localhost:7687",
                    "NEO4J_USERNAME": "neo4j",
                    "NEO4J_PASSWORD": "password",
                    "NEO4J_DATABASE": "Neo4j"
                }
            }
        })
        tools = await client.get_tools()
        # Filter out get_neo4j_schema tool
        tools = [tool for tool in tools if tool.name != 'get_neo4j_schema']
        return tools
    
    tools = run_async_task(get_tools())
    
    # Initialize Gemini
    gemini = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        temperature=0
    )
    
    # Create agent
    agent = create_react_agent(gemini, tools)
    
    return agent, system_prompt

# Async function to query the agent
async def query_agent(agent, messages):
    steps = []
    final_response = None

    try:
        async for event in agent.astream(
            {"messages": messages},
            config={"recursion_limit": 50},
            stream_mode="values"
        ):
            if "messages" in event:
                event_messages = event["messages"]
                if event_messages:
                    last_msg = event_messages[-1]
                    msg_type = last_msg.__class__.__name__

                    if msg_type == "AIMessage":
                        tool_calls = getattr(last_msg, 'tool_calls', [])
                        content = getattr(last_msg, 'content', '')

                        if tool_calls:
                            for tool_call in tool_calls:
                                steps.append({
                                    'type': 'tool_call',
                                    'tool': tool_call.get('name', 'unknown'),
                                    'args': tool_call.get('args', {})
                                })
                        elif content:
                            steps.append({
                                'type': 'final_answer',
                                'content': content
                            })

                    elif msg_type == "ToolMessage":
                        content = getattr(last_msg, 'content', '')
                        tool_name = getattr(last_msg, 'name', 'unknown')

                        try:
                            result = json.loads(content)
                        except json.JSONDecodeError:
                            result = content

                        steps.append({
                            'type': 'tool_result',
                            'tool': tool_name,
                            'result': result
                        })

            final_response = event

    except Exception as e:
        # Log error instead of failing silently
        st.error(f"Error during agent execution: {e}")
        return None, steps

    if final_response:
        response_data = final_response.get("agent", final_response) if "agent" in final_response else final_response
        cleaned = extract_response(response_data)
        return cleaned, steps

    return None, steps


def render_sidebar() -> str:
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.info("Connected to Neo4j database")

        st.markdown("### Authentication")
        entered_key = st.text_input(
            "Gemini API key",
            value=st.session_state.user_api_key,
            type="password",
            help="Provide your own Google AI Studio key. Stored only for this session."
        )
        st.session_state.user_api_key = entered_key.strip()

        if st.button("Clear stored API key"):
            st.session_state.user_api_key = ""
            st.session_state.agent = None
            st.session_state.agent_key = None
            st.session_state.system_prompt = None
            initialize_agent.clear()
            st.rerun()

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("### Memory")
        st.session_state.memory_enabled = st.checkbox(
            "Remember previous answers",
            value=st.session_state.memory_enabled
        )
        if st.session_state.memory_enabled:
            current_depth = st.session_state.memory_depth if 1 <= st.session_state.memory_depth <= 5 else 1
            depth_choice = st.slider(
                "How many previous answers?",
                min_value=1,
                max_value=5,
                value=current_depth,
                step=1
            )
            st.session_state.memory_depth = depth_choice
        else:
            st.session_state.memory_depth = 0

        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app uses Google's API, MCP Neo4j server and Langgraph to query IYP.")

    current_key = st.session_state.user_api_key.strip()
    st.session_state.user_api_key = current_key
    return current_key


current_api_key = render_sidebar()

if not current_api_key:
    st.info("Enter your Gemini API key in the sidebar to start chatting.")
    st.stop()

if st.session_state.agent_key and st.session_state.agent_key != current_api_key:
    st.session_state.agent = None

# Initialize agent
with st.spinner("Initializing agent..."):
    if st.session_state.agent is None:
        st.session_state.agent, st.session_state.system_prompt = initialize_agent(current_api_key)
        st.session_state.agent_key = current_api_key
        st.success("Agent initialized successfully!")

render_user_guidance()




# Chat interface
st.markdown("---")

# Display chat history
for chat_idx, chat in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.write(chat['question'])
    
    with st.chat_message("assistant"):
        st.write(chat['answer'])
        
        # Show Cypher queries in expander
        if chat.get('cypher_queries'):
            with st.expander(f"📊 View {len(chat['cypher_queries'])} Cypher Query/Queries"):
                for idx, query_data in enumerate(chat['cypher_queries'], 1):
                    st.code(query_data['query'], language='cypher')
                    
                    # Find matching result
                    matching_result = next(
                        (r for r in chat.get('cypher_results', []) if r['tool_call_id'] == query_data['tool_call_id']),
                        None
                    )
                    
                    if matching_result:
                        st.json(matching_result['result'])
                    st.markdown("---")

    delete_key = f"delete_chat_{chat_idx}"
    if st.button("🗑️ Delete this query", key=delete_key, type="primary"):
        st.session_state.chat_history.pop(chat_idx)
        st.rerun()

# User input
user_query = st.chat_input("Ask a question about your Neo4j database...")

if user_query:
    # Add user message to chat
    with st.chat_message("user"):
        st.write(user_query)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking..."):
            conversation_messages = build_conversation_messages(
                st.session_state.system_prompt,
                user_query
            )
            # Run async query using proper async handling
            try:
                response_data, steps = run_async_task(
                    query_agent(st.session_state.agent, conversation_messages)
                )
            except Exception as exc:
                st.error(f"Error while running agent: {exc}")
                response_data, steps = None, []
            
            if response_data:
                # Display reasoning steps in expander
                if steps:
                    with st.expander("🧠 View Reasoning Steps", expanded=False):
                        for idx, step in enumerate(steps, 1):
                            if step['type'] == 'tool_call':
                                st.markdown(f"**Step {idx}: Tool Call**")
                                st.write(f"Tool: `{step['tool']}`")
                                st.json(step['args'])
                            elif step['type'] == 'tool_result':
                                st.markdown(f"**Step {idx}: Tool Result**")
                                st.write(f"Tool: `{step['tool']}`")
                                st.json(step['result'])
                            elif step['type'] == 'final_answer':
                                st.markdown(f"**Step {idx}: Final Answer**")
                                st.write(step['content'])
                            st.markdown("---")
                
                # Display answer
                st.write(response_data['llm_answer'])
                
                # Show Cypher queries
                if response_data.get('cypher_queries'):
                    with st.expander(f"📊 View {len(response_data['cypher_queries'])} Cypher Query/Queries"):
                        for idx, query_data in enumerate(response_data['cypher_queries'], 1):
                            st.markdown(f"**Query {idx}:**")
                            st.code(query_data['query'], language='cypher')
                            
                            # Find matching result
                            matching_result = next(
                                (r for r in response_data.get('cypher_results', []) if r['tool_call_id'] == query_data['tool_call_id']),
                                None
                            )
                            
                            if matching_result:
                                st.markdown("**Result:**")
                                st.json(matching_result['result'])
                            st.markdown("---")
                
                # Add to chat history
                st.session_state.chat_history.append({
                    'question': user_query,
                    'answer': response_data['llm_answer'],
                    'cypher_queries': response_data.get('cypher_queries', []),
                    'cypher_results': response_data.get('cypher_results', [])
                })
            else:
                st.error("Failed to get response from agent")
