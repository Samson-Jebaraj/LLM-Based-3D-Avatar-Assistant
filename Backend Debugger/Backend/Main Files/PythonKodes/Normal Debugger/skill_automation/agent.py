"""Home Assistant Generative Agent implementation."""
from __future__ import annotations

import logging
from typing import Any, Optional
from pathlib import Path

from langchain.agents import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from ulid import ULID

from .graph import workflow, State
from .const import (
    AGENT_SYSTEM_PROMPT,
    DEFAULT_MEMORY_KEY,
)

LOGGER = logging.getLogger(__name__)

class HomeAgent:
    """Home Assistant Generative Agent implementation."""

    def __init__(
        self,
        chat_model: Any,
        vlm_model: Any,
        ha_llm_api: Any,
        user_id: str,
        options: dict[str, Any],
    ) -> None:
        """Initialize the agent."""
        self.chat_model = chat_model
        self.vlm_model = vlm_model
        self.ha_llm_api = ha_llm_api
        self.user_id = user_id
        self.options = options

        # Initialize tools
        from .tools import upsert_memory, add_automation
        
        self.tools = [
            Tool(
                func=upsert_memory,
                name="upsert_memory",
                description="Store or update a memory in the database"
            ),
            Tool(
                func=add_automation,
                name="add_automation",
                description="Add an automation to Home Assistant"
            ),
        ]

        # Configure the workflow
        self.config = RunnableConfig(
            configurable={
                "chat_model": self.chat_model,
                "vlm_model": self.vlm_model,
                "ha_llm_api": self.ha_llm_api,
                "user_id": self.user_id,
                "options": self.options,
                "langchain_tools": {tool.name.lower(): tool for tool in self.tools},
                "prompt": AGENT_SYSTEM_PROMPT,
            }
        )

        # Initialize the graph with store
        self.graph = workflow.compile()

    async def ainvoke(
        self,
        message: str,
        *,
        conversation_id: Optional[str] = None,
    ) -> str:
        """Invoke the agent asynchronously."""
        try:
            # Create initial state
            state = State(
                messages=[HumanMessage(content=message)],
                summary=""
            )

            # Run the graph
            result = await self.graph.ainvoke(
                state,
                config=self.config,
            )

            # Extract the final response
            final_message = result["messages"][-1]
            return final_message.content

        except Exception as e:
            LOGGER.error("Error in agent invocation: %s", e, exc_info=True)
            raise

    async def astart(self) -> None:
        """Start the agent."""
        pass
        

    async def astop(self) -> None:
        """Stop the agent."""
        pass