"""
Multi-agent orchestration with LangGraph supervisor pattern.
"""

from typing import TypedDict, Annotated, Literal, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
import operator


class AgentState(TypedDict):
    """Typed state that flows through the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_agent: str
    iteration_count: int
    context: dict


def create_supervisor_chain(llm: ChatAnthropic, agent_names: list[str]):
    """Supervisor decides which agent handles the next step."""
    
    system_prompt = f"""You are a supervisor managing these agents: {agent_names}.
    
    Given the conversation and current state, decide which agent should act next.
    If the task is complete, respond with FINISH.
    
    Respond with ONLY the agent name or FINISH."""
    
    def supervisor(state: AgentState) -> dict:
        messages = state["messages"]
        
        response = llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current state: {messages[-3:]}"}
        ])
        
        next_agent = response.content.strip()
        
        return {
            "next_agent": next_agent,
            "iteration_count": state["iteration_count"] + 1
        }
    
    return supervisor


def build_multi_agent_graph(
    llm: ChatAnthropic,
    researcher_tools: list,
    analyst_tools: list,
    executor_tools: list,
    db_connection_string: str
) -> StateGraph:
    """Builds the complete multi-agent graph with checkpointing."""
    
    # Create specialized agents
    researcher = create_react_agent(llm, researcher_tools)
    analyst = create_react_agent(llm, analyst_tools)
    executor = create_react_agent(llm, executor_tools)
    
    supervisor = create_supervisor_chain(
        llm, 
        ["researcher", "analyst", "executor"]
    )
    
    # Build the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("researcher", researcher)
    workflow.add_node("analyst", analyst)
    workflow.add_node("executor", executor)
    
    # Routing function
    def route_to_agent(state: AgentState) -> str:
        # Prevent infinite loops
        if state["iteration_count"] > 10:
            return "__end__"
        
        next_agent = state["next_agent"]
        
        if next_agent == "FINISH":
            return "__end__"
        elif next_agent in ["researcher", "analyst", "executor"]:
            return next_agent
        else:
            return "__end__"
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges("supervisor", route_to_agent)
    
    # All agents return to supervisor
    workflow.add_edge("researcher", "supervisor")
    workflow.add_edge("analyst", "supervisor")
    workflow.add_edge("executor", "supervisor")
    
    # PostgreSQL checkpointing
    checkpointer = PostgresSaver.from_conn_string(db_connection_string)
    checkpointer.setup()  # Create required tables
    
    return workflow.compile(checkpointer=checkpointer)


async def run_agent_workflow(
    graph: StateGraph,
    user_input: str,
    thread_id: str
) -> dict:
    """Run the workflow with automatic state persistence."""
    
    config = {"configurable": {"thread_id": thread_id}}
    
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "next_agent": "",
        "iteration_count": 0,
        "context": {}
    }
    
    result = await graph.ainvoke(initial_state, config)
    
    return result
