import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
import importlib.util

from langchain import memory
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import LLMChain, LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import create_react_agent, AgentExecutor, Tool, tool
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore

import os
from dotenv import load_dotenv

import base64
import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any
import aiofiles
from collections.abc import Mapping

from .const import (
    BLUEPRINT_NAME,
    CONF_VISION_MODEL_TEMPERATURE,
    CONF_VISION_MODEL_TOP_P,
    CONF_VLM,
    EVENT_AUTOMATION_REGISTERED,
    RECOMMENDED_VISION_MODEL_TEMPERATURE,
    RECOMMENDED_VISION_MODEL_TOP_P,
    RECOMMENDED_VLM,
    VISION_MODEL_IMAGE_HEIGHT,
    VISION_MODEL_IMAGE_WIDTH,
    VISION_MODEL_SYSTEM_PROMPT,
    VISION_MODEL_USER_KW_PROMPT,
    VISION_MODEL_USER_PROMPT,
    VLM_NUM_PREDICT,
)
from .graph import workflow 
from datetime import datetime
import dateutil.parser
from zoneinfo import ZoneInfo
import yaml
from ulid import ULID

DEFAULT_AUTOMATION_PATH = "automations.yaml"

try:
    load_dotenv(dotenv_path="./API_KEYS.env")
    wolfram_api_key = os.environ.get("WOLFRAM_ALPHA_API_KEY")
    base_data_path = Path("./BASE_DATA.py")

    def as_utc(dattim: str, default: datetime, error_message: str) -> datetime:
        """
        Convert a string representing a datetime into a datetime.datetime in UTC.
        Args:
            dattim: String representing a datetime.
            default: datetime.datetime to use as default.
            error_message: Message to raise in case of error.
        Raises:
            Exception if datetime cannot be parsed.
        Returns:
            A datetime.datetime of the string in UTC.
        """
        if dattim is None:
            return default
    
        try:
            parsed_datetime = dateutil.parser.parse(dattim)
            if parsed_datetime.tzinfo is None:
                parsed_datetime = parsed_datetime.replace(tzinfo=ZoneInfo("UTC"))
            
            return parsed_datetime.astimezone(ZoneInfo("UTC"))
        
        except (ValueError, TypeError):
            raise Exception(error_message)



    # async def _get_camera_image(hass: HomeAssistant, camera_name: str) -> bytes:
    #     """Get an image from a given camera."""
    #     camera_entity_id: str = f"camera.{camera_name.lower()}"
    #     try:
    #         image = await camera.async_get_image(
    #             hass=hass,
    #             entity_id=camera_entity_id,
    #             width=VISION_MODEL_IMAGE_WIDTH,
    #             height=VISION_MODEL_IMAGE_HEIGHT
    #         )
    #     except HomeAssistantError as err:
    #         LOGGER.error(
    #             "Error getting image from camera '%s' with error: %s",
    #             camera_entity_id, err
    #         )

    #     return image.content

    # async def _analyze_image(
    #         vlm_model: ChatOllama,
    #         options: dict[str, Any] | MappingProxyType[str, Any],
    #         image: bytes,
    #         detection_keywords: list[str] | None = None
    #     ) -> str:
    #     """Analyze an image."""
    #     encoded_image = base64.b64encode(image).decode("utf-8")

    #     def prompt_func(data: dict[str, Any]) -> list[AnyMessage]:
    #         system = data["system"]
    #         text = data["text"]
    #         image = data["image"]

    #         text_part = {"type": "text", "text": text}
    #         image_part = {
    #             "type": "image_url",
    #             "image_url": {"url": f"data:image/jpeg;base64,{image}"},
    #         }

    #         content_parts = []
    #         content_parts.append(text_part)
    #         content_parts.append(image_part)

    #         return [SystemMessage(content=system), HumanMessage(content=content_parts)]

    #     model = vlm_model
    #     model_with_config = model.with_config(
    #         config={
    #             "model": options.get(
    #                 CONF_VLM,
    #                 RECOMMENDED_VLM,
    #             ),
    #             "temperature": options.get(
    #                 CONF_VISION_MODEL_TEMPERATURE,
    #                 RECOMMENDED_VISION_MODEL_TEMPERATURE,
    #             ),
    #             "top_p": options.get(
    #                 CONF_VISION_MODEL_TOP_P,
    #                 RECOMMENDED_VISION_MODEL_TOP_P,
    #             ),
    #             "num_predict": VLM_NUM_PREDICT,
    #         }
    #     )

    #     chain = prompt_func | model_with_config

    #     if detection_keywords is not None:
    #         prompt = f"{VISION_MODEL_USER_KW_PROMPT} {' or '.join(detection_keywords):}"
    #     else:
    #         prompt = VISION_MODEL_USER_PROMPT

    #     try:
    #         response =  await chain.ainvoke(
    #             {
    #                 "system": VISION_MODEL_SYSTEM_PROMPT,
    #                 "text": prompt,
    #                 "image": encoded_image
    #             }
    #         )
    #     except Exception as err: #TODO: add validation error handling and retry prompt
    #         LOGGER.error("Error analyzing image %s", err)

    #     return response

    # @tool(parse_docstring=False)
    # async def get_and_analyze_camera_image( # noqa: D417
    #         camera_name: str,
    #         detection_keywords: list[str] | None = None,
    #         *,
    #         # Hide these arguments from the model.
    #         config: Annotated[RunnableConfig, InjectedToolArg()],
    #     ) -> str:
    #     """
    #     Get a camera image and perform scene analysis on it.

    #     Args:
    #         camera_name: Name of the camera for scene analysis.
    #         detection_keywords: Specific objects to look for in image, if any.
    #             For example, If user says "check the front porch camera for
    #             boxes and dogs", detection_keywords would be ["boxes", "dogs"].

    #     """
    #     hass = config["configurable"]["hass"]
    #     vlm_model = config["configurable"]["vlm_model"]
    #     options = config["configurable"]["options"]
    #     image = await _get_camera_image(hass, camera_name)
    #     return await _analyze_image(vlm_model, options, image, detection_keywords)

    @tool(parse_docstring=False)
    async def upsert_memory(
        content: str,
        context: str,
        *,
        memory_id: ULID | None = None,
        config: Annotated[RunnableConfig, InjectedToolArg()],
        store: Annotated[BaseStore, InjectedStore()],
    ) -> str:
        """
        Upsert a memory in the database.

        If a memory conflicts with an existing one, then just UPDATE the
        existing one by passing in memory_id - don't create two memories
        that are the same. If the user corrects a memory, UPDATE it.

        Args:
            content: The main content of the memory. For example:
                "User expressed interest in learning about French."
            context: Additional context for the memory. For example:
                "This was mentioned while discussing career options in Europe."
            memory_id: ONLY PROVIDE IF UPDATING AN EXISTING MEMORY.
                The memory to overwrite

        Returns:
            A string containing the stored memory id.

        """
        mem_id = memory_id or ulid.ulid_now()
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
        config: Annotated[RunnableConfig, InjectedToolArg()]
    ) -> str:
        """
        Add an automation to Home Assistant.
        You are provided a Home Assistant blueprint as part of this tool if you need it.
        You MUST ONLY use the blueprint to create automations that involve camera image
        analysis. You MUST generate Home Assistant automation yaml for everything else.
        If using the blueprint you MUST provide the arguments "time_pattern" and "message"
        and DO NOT provide the argument "automation_yaml".
        Args:
            automation_yaml: A Home Assistant automation in valid yaml format.
            time_pattern: Cron-like time pattern (e.g., /30 for "every 30 mins").
            message: Image analysis prompt (e.g.,"check the front porch camera for boxes")
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
        ha_automation_config = {"id": str(ulid.ulid_now())}
    
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
                if content.strip():  # Only parse if file is not empty
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




except Exception as e:
    error_msg = f'Initialization error: {str(e)}\n{traceback.format_exc()}'
    print(f"[KODEZ_ERROR]{error_msg}")
    sys.stdout.flush()