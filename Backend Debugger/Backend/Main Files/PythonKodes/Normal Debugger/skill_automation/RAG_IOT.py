import asyncio
from pathlib import Path
import os
import sys
import importlib.util
from dotenv import load_dotenv

async def main():
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
    user_request = input("Enter your automation request: ")
    from .agent import HomeAgent
    
    agent = HomeAgent(
        chat_model=base_data.llm,
        vlm_model=None,
        ha_llm_api=None,
        user_id="user123",
        options={"single_request": True}
    )
    
    try:
        # Process single request
        print("Processing request...")
        response = await agent.ainvoke(user_request)
        print(f"\nResponse: {response}")
        
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        
    finally:
        await agent.astop()

if __name__ == "__main__":
    asyncio.run(main())