"""UI preview with a scripted LLM, without an API key: uv run python -m tests.dev_server"""

import uvicorn

from product_card.api import create_app
from product_card.config import Settings

from .fakes import ScriptedLLM, draft_payload

if __name__ == "__main__":
    print("DEV MODE: answers are scripted, the LLM is not called.")
    settings = Settings(_env_file=None, rate_limit_per_minute=1000)
    uvicorn.run(
        create_app(settings, ScriptedLLM([draft_payload()])), port=8765, log_level="warning"
    )
