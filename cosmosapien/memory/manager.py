"""Memory manager for conversation history."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.models import ChatMessage


class MemoryManager:
    """Manages conversation memory and history."""

    def __init__(self, memory_path: str = "~/.cosmosapien/memory"):
        self.memory_path = Path(memory_path).expanduser()
        self.memory_path.mkdir(parents=True, exist_ok=True)

    def save_conversation(
        self,
        session_id: str,
        messages: List[ChatMessage],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save a conversation to memory."""
        conversation_data = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "messages": [msg.dict() for msg in messages],
            "metadata": metadata or {},
        }

        file_path = self.memory_path / "{session_id}.json"
        with open(file_path, "w") as f:
            json.dump(conversation_data, f, indent=2)

    def load_conversation(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a conversation from memory."""
        file_path = self.memory_path / "{session_id}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # Convert messages back to ChatMessage objects
            data["messages"] = [ChatMessage(**msg) for msg in data["messages"]]
            return data
        except Exception:
            return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all conversation sessions."""
        sessions = []

        for file_path in self.memory_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                sessions.append(
                    {
                        "session_id": data["session_id"],
                        "timestamp": data["timestamp"],
                        "message_count": len(data["messages"]),
                        "metadata": data.get("metadata", {}),
                    }
                )
            except Exception:
                continue

        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x["timestamp"], reverse=True)
        return sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a conversation session."""
        file_path = self.memory_path / "{session_id}.json"

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def clear_all(self) -> None:
        """Clear all conversation history."""
        for file_path in self.memory_path.glob("*.json"):
            file_path.unlink()
