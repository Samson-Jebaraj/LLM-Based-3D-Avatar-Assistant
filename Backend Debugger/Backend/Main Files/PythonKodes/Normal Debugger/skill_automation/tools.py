from typing import Annotated, Any
from pathlib import Path
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedStore
from langchain_core.tools import InjectedToolArg
from langgraph.store.base import BaseStore
from ulid import ULID
import yaml
import aiofiles
from collections.abc import Mapping
from langchain_core.tools import tool

# Constants
BLUEPRINT_NAME = "camera_analysis_blueprint"
EVENT_AUTOMATION_REGISTERED = "automation_registered"
DEFAULT_AUTOMATION_PATH = "automations.yaml"

@tool(parse_docstring=False)
async def upsert_memory(
    content: str,
    context: str,
    *,
    memory_id: ULID | None = None,
    config: Annotated[RunnableConfig, InjectedToolArg],
    store: Annotated[BaseStore, InjectedStore],
) -> str:
    """
    Upsert a memory in the database.
    
    Args:
        content: The main content of the memory
        context: Additional context for the memory
        memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY
    """
    mem_id = memory_id or ULID()
    await store.aput(
        namespace=(config["configurable"]["user_id"], "memories"),
        key=str(mem_id),
        value={"content": content, "context": context},
    )
    return f"Stored memory {mem_id}"

@tool(parse_docstring=False)
async def add_automation(
    automation_yaml: str | None = None,
    time_pattern: str | None = None,
    message: str | None = None,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """
    Add an automation to Home Assistant.
    """
    hass = config["configurable"]["hass"]
    config_dir = Path(hass.config.config_dir)
    automation_path = config_dir / DEFAULT_AUTOMATION_PATH

    if time_pattern is not None and message is not None:
        automation_data = {
            "alias": message,
            "description": f"Created with blueprint {BLUEPRINT_NAME}.",
            "use_blueprint": {
                "path": BLUEPRINT_NAME,
                "input": {
                    "time_pattern": time_pattern,
                    "message": message,
                }
            }
        }
        automation_yaml = yaml.dump(automation_data)

    automation_parsed = yaml.safe_load(automation_yaml)
    ha_automation_config = {"id": str(ULID())}

    if isinstance(automation_parsed, list):
        ha_automation_config.update(automation_parsed[0])
    elif isinstance(automation_parsed, Mapping):
        ha_automation_config.update(automation_parsed)
    else:
        raise ValueError("Invalid automation configuration format")

    config_dir.mkdir(parents=True, exist_ok=True)
    existing_automations = []
    
    if automation_path.exists():
        async with aiofiles.open(automation_path, encoding="utf-8") as f:
            content = await f.read()
            if content.strip():
                existing_automations = yaml.safe_load(content) or []
                if not isinstance(existing_automations, list):
                    existing_automations = [existing_automations] if existing_automations else []

    existing_automations.append(ha_automation_config)
    async with aiofiles.open(automation_path, "w", encoding="utf-8") as f:
        await f.write(yaml.dump(existing_automations, allow_unicode=True, sort_keys=False))

    hass.bus.async_fire(
        EVENT_AUTOMATION_REGISTERED,
        {
            "automation_config": ha_automation_config,
            "raw_config": yaml.dump([ha_automation_config], allow_unicode=True, sort_keys=False),
        },
    )

    return f"Added automation {ha_automation_config['id']}"