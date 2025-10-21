from common import *
from init_llm import *

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.core.agent.workflow import AgentWorkflow  # orchestrator

from typing import List
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

# Define structured outputs
class WebSearchResult(BaseModel):
    query: str = Field(..., description="Search query string")
    results: List[str] = Field(..., description="List of results from the web")


class KBSearchResult(BaseModel):
    query: str = Field(..., description="Knowledge base query string")
    answer: str = Field(..., description="Answer from ChromaDB knowledge base")


# Functions with structured return
def search_web(query: str) -> WebSearchResult:
    return WebSearchResult(
        query=query,
        results=[f"Result: 1"] #simulate web search results
    )


def search_knowledgebase(query: str) -> KBSearchResult:
    return KBSearchResult(
        query=query,
        answer=f"Result: 13" #any db call
    )


async def run_agent(query: str) -> str:
    ensure_llamaindex_settings()
    web_tool = FunctionTool.from_defaults(fn=search_web)
    kb_tool  = FunctionTool.from_defaults(fn=search_knowledgebase)
    wf = AgentWorkflow.from_tools_or_functions([web_tool, kb_tool])
    result = await wf.run(query)
    return str(result)  

async def run_agent_sync(query: str) -> str:
    ensure_llamaindex_settings()
    agent = ReActAgent(tools=[search_web, search_knowledgebase], llm=Settings.llm)

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
