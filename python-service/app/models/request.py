from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Literal

class ChatRequest(BaseModel):
    session_id: Optional[str] = "default"
    user_id: Optional[str] = None
    message: str
    file: Optional[str] = None
    # keep compatible with RAG payloads and future agent extensions
    use_graph: Optional[bool] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    model_provider: Optional[Literal["qwen", "deepseek"]] = None
    model_name: Optional[str] = None
    # Agent-only: "fast" merges LLM steps for latency; "full" runs the legacy multi-step agents.
    agent_pipeline: Optional[Literal["fast", "full"]] = "fast"
