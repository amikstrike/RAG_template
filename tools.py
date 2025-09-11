from common import *
from init_llm import *

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow  # orchestrator


def search_Web(query: str) -> str:
    """Search web for additional information and returns the result"""
    return f"Web result 165333 for: {query}"

def search_KnowledgeBase(query: str) -> str:
    """Search knowledge base for additional information and returns the result"""
    return f"Knowledge Base result 4444 for: {query}"

async def run_agent(query: str) -> str:
    ensure_llamaindex_agent_settings()
    web_tool = FunctionTool.from_defaults(fn=search_Web)
    kb_tool  = FunctionTool.from_defaults(fn=search_KnowledgeBase)
    wf = AgentWorkflow.from_tools_or_functions([web_tool, kb_tool])
    result = await wf.run(query)
    return str(result)  

async def run_agent_sync(query: str) -> str:
    ensure_llamaindex_agent_settings()
    agent = ReActAgent(tools=[search_Web, search_KnowledgeBase], llm=Settings.llm)

    # Create a context to store the conversation history/session state
    ctx = Context(agent)
        
    from llama_index.core.agent.workflow import AgentStream, ToolCallResult

    handler = agent.run(query, ctx=ctx)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCallResult):
             print(f"\nCall {ev.tool_name} with {ev.tool_kwargs}\nReturned: {ev.tool_output}")
        elif isinstance(ev, AgentStream):
            print(f"{ev.delta}", end="", flush=True)
        elif ev is not None:
            print(f"{ev}", end="", flush=True)

    response = await handler
    return str(response)
