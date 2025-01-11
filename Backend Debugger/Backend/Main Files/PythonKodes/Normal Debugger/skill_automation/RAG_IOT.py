import asyncio
from langchain_community.chat_models import ChatOllama
from .store import MemoryStore
from .agent import HomeAgent
import os
import sys
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

async def main():
    # Initialize components
    current_dir = Path(__file__).parent
    project_root = current_dir.parent
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
    chat_model = base_data.llm
    vlm_model = base_data.llm
    
    # Create agent
    agent = HomeAgent(
        chat_model=chat_model,
        vlm_model=None,
        ha_llm_api=None,
        user_id="user123",
        options={}
    )
    
    # Start the agent
    await agent.astart()
    
    try:
        # Test the agent
        response = await agent.ainvoke("Create an automation to check the front door camera every 30 minutes")
        print(f"Agent response: {response}")
        
        response = await agent.ainvoke("What's the last thing we discussed?")
        print(f"Agent response: {response}")
        
    finally:
        # Clean up
        await agent.astop()

if __name__ == "__main__":
    asyncio.run(main())