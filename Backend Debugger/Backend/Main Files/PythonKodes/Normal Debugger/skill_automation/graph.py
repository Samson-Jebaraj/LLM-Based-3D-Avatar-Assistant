"""Langgraph graphs for Home Generative Agent."""
from __future__ import annotations  # noqa: I001

import copy
import json
import logging
from typing import Literal
import os
import sys
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from langchain_core.runnables import RunnableConfig  # noqa: TCH002
from langgraph.store.base import BaseStore  # noqa: TCH002
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import ValidationError
from pathlib import Path
import importlib.util
from .const import (
    CONF_SUMMARIZATION_MODEL_TEMPERATURE,
    CONF_SUMMARIZATION_MODEL_TOP_P,
    CONF_VLM,
    CONTEXT_MAX_MESSAGES,
    CONTEXT_SUMMARIZE_THRESHOLD,
    EMBEDDING_MODEL_PROMPT_TEMPLATE,
    RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
    RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
    RECOMMENDED_VLM,
    SUMMARY_INITIAL_PROMPT,
    SUMMARY_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    TOOL_CALL_ERROR_TEMPLATE,
    VLM_NUM_PREDICT,
)
from dotenv import load_dotenv
import time
import random
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

LOGGER = logging.getLogger(__name__)

current_dir = Path(__file__).parent
project_root = current_dir.parent
        
# Load environment variables
env_path = project_root / "API_KEYS.env"
if not env_path.exists():
    raise FileNotFoundError(f"Environment file not found at {env_path}")
        
load_dotenv(dotenv_path=str(env_path))
wolfram_api_key = os.environ.get("WOLFRAM_ALPHA_API_KEY")
        
if not wolfram_api_key:
    raise ValueError("WOLFRAM_ALPHA_API_KEY not found in environment variables")
            
base_data_path = project_root / "BASE_DATA.py"
if not base_data_path.exists():
    raise FileNotFoundError(f"BASE_DATA.py not found at {base_data_path}")
            
sys.path.insert(0, str(project_root))
        
try:
    spec = importlib.util.spec_from_file_location("BASE_DATA", base_data_path)
    base_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base_data)

except Exception as e:
    raise ImportError(f"Failed to import BASE_DATA: {str(e)}")

class State(MessagesState):
    """Extend the MessagesState to include a summary key."""

    summary: str

# Add retry decorator for model calls
@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3)
)
async def _call_model(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[BaseMessage]]:
    """Coroutine to call the model with retry logic."""
    try:
        # Add small random delay to avoid burst requests
        await asyncio.sleep(random.uniform(0.5, 2))
        
        model = config["configurable"]["chat_model"]
        prompt = config["configurable"]["prompt"]
        user_id = config["configurable"]["user_id"]

        # Initialize empty memories message
        mems_message = ""
        
        # Only attempt to retrieve memories if store is available
        if store is not None:
            # Retrieve most recent or search for most relevant memories for context.
            # Use semantic search if the last message was from the user.
            msg = state["messages"][-1]
            query_prompt = EMBEDDING_MODEL_PROMPT_TEMPLATE.format(
                query=msg.content
            ) if isinstance(msg, HumanMessage) else None
            
            try:
                mems = await store.asearch(
                    (user_id, "memories"),
                    query=query_prompt,
                    limit=10
                )
                formatted_mems = "\n".join(f"[{mem.key}]: {mem.value}" for mem in mems)
                mems_message = f"\n<memories>\n{formatted_mems}\n</memories>" if formatted_mems else ""
            except Exception as e:
                LOGGER.warning("Failed to retrieve memories: %s", str(e))

        # Rest of the function remains the same
        summary = state.get("summary", "")
        summary_message = f"\nSummary of conversation earlier: {summary}" if summary else ""

        messages = [SystemMessage(
            content=(prompt + mems_message + summary_message)
        )] + state["messages"]

        LOGGER.debug("Model call messages: %s", messages)
        LOGGER.debug("Model call messages length: %s", len(messages))

        response = await model.ainvoke(messages)
        return {"messages": response}
        
    except Exception as e:
        if "quota" in str(e).lower():
            LOGGER.warning("Quota limit reached, retrying after delay: %s", str(e))
            raise  # This will trigger the retry
        raise  # Re-raise other exceptions

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(3)
)
async def _summarize_and_trim(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[AnyMessage]]:
    """Coroutine to summarize and trim message history."""
    summary = state.get("summary", "")

    if summary:
        summary_message = SUMMARY_PROMPT_TEMPLATE.format(summary=summary)
    else:
        summary_message = SUMMARY_INITIAL_PROMPT

    messages = (
        [SystemMessage(content=SUMMARY_SYSTEM_PROMPT)] +
        state["messages"] +
        [HumanMessage(content=summary_message)]
    )

    model = config["configurable"]["vlm_model"]
    options = config["configurable"]["options"]
    model_with_config = model.with_config(
        config={
            "model": options.get(
                CONF_VLM,
                RECOMMENDED_VLM,
            ),
            "temperature": options.get(
                CONF_SUMMARIZATION_MODEL_TEMPERATURE,
                RECOMMENDED_SUMMARIZATION_MODEL_TEMPERATURE,
            ),
            "top_p": options.get(
                CONF_SUMMARIZATION_MODEL_TOP_P,
                RECOMMENDED_SUMMARIZATION_MODEL_TOP_P,
            ),
            "num_predict": VLM_NUM_PREDICT,
        }
    )

    LOGGER.debug("Summary messages: %s", messages)
    try:
        await asyncio.sleep(random.uniform(0.5, 2))
        response = await model_with_config.ainvoke(messages)
    except Exception as e:
        if "quota" in str(e).lower():
            LOGGER.warning("Quota limit reached, retrying after delay: %s", str(e))
            raise
        raise

    # Trim message history to manage context window length.
    trimmed_messages = trim_messages(
        messages=state["messages"],
        token_counter=len,
        max_tokens=CONTEXT_MAX_MESSAGES,
        strategy="last",
        start_on="human",
        include_system=True,
    )
    messages_to_remove = [m for m in state["messages"] if m not in trimmed_messages]
    LOGGER.debug("Messages to remove: %s", messages_to_remove)
    remove_messages = [RemoveMessage(id=m.id) for m in messages_to_remove]

    return {"summary": response.content, "messages": remove_messages}

async def _call_tools(
        state: State, config: RunnableConfig, *, store: BaseStore
    ) -> dict[str, list[ToolMessage]]:
    """Coroutine to call Home Assistant or langchain LLM tools."""
    # Tool calls will be the last message in state.
    tool_calls = state["messages"][-1].tool_calls

    langchain_tools = config["configurable"]["langchain_tools"]
    ha_llm_api = config["configurable"]["ha_llm_api"]

    tool_responses: list[ToolMessage] = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        LOGGER.debug(
            "Tool call: %s(%s)", tool_name, tool_args
        )

        def _handle_tool_error(err:str, name:str, tid:str) -> ToolMessage:
            return ToolMessage(
                content=TOOL_CALL_ERROR_TEMPLATE.format(error=err),
                name=name,
                tool_call_id=tid,
                status="error",
            )

        # A langchain tool was called.
        if tool_name in langchain_tools:
            lc_tool = langchain_tools[tool_name.lower()]

            tool_call_copy = copy.deepcopy(tool_call)
            tool_call_copy["args"].update(
                {
                    "store": store,
                    "config": config,
                }
            )

            try:
                tool_response = await lc_tool.ainvoke(tool_call_copy)
            except (Exception, ValidationError) as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])
        else:
            tool_input = base_data.llm.ToolInput(
                tool_name=tool_name,
                tool_args=tool_args,
            )

            try:
                response = await ha_llm_api.async_call_tool(tool_input)

                tool_response = ToolMessage(
                    content=json.dumps(response),
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            except Exception as e:
                tool_response = _handle_tool_error(repr(e), tool_name, tool_call["id"])

        LOGGER.debug("Tool response: %s", tool_response)
        tool_responses.append(tool_response)
    return {"messages": tool_responses}

def _should_continue(
        state: State
    ) -> Literal["action", "summarize_and_trim", "__end__"]:
    """Return the next node in graph to execute."""
    messages = state["messages"]

    if messages[-1].tool_calls:
        return "action"

    if len(messages) > CONTEXT_SUMMARIZE_THRESHOLD:
        LOGGER.debug("Summarizing conversation")
        return "summarize_and_trim"

    return "__end__"

# Define a new graph
workflow = StateGraph(State)

# Define nodes.
workflow.add_node("agent", _call_model)
workflow.add_node("action", _call_tools)
workflow.add_node("summarize_and_trim", _summarize_and_trim)

# Define edges.
workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", _should_continue)
workflow.add_edge("action", "agent")
workflow.add_edge("summarize_and_trim", END)
