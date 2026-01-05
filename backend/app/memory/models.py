"""Data models for conversation memory."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class MessageRole(str, Enum):
    """Role of a message in conversation."""
    
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(BaseModel):
    """A single message in a conversation."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional metadata
    sources: Optional[list[dict]] = None  # Retrieved sources for this message
    tokens_used: Optional[int] = None
    latency_ms: Optional[int] = None
    
    class Config:
        use_enum_values = True


class Conversation(BaseModel):
    """A conversation session with message history."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    document_id: Optional[str] = None  # Scope conversation to a document
    title: Optional[str] = None
    
    messages: list[Message] = Field(default_factory=list)
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Conversation state
    is_active: bool = True
    summary: Optional[str] = None  # Auto-generated summary for long conversations
    
    class Config:
        use_enum_values = True
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        **kwargs,
    ) -> Message:
        """Add a message to the conversation."""
        message = Message(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
        return message
    
    def get_context_window(
        self,
        max_messages: int = 10,
        max_tokens: int = 4000,
    ) -> list[Message]:
        """
        Get recent messages that fit within context limits.
        
        Args:
            max_messages: Maximum number of messages
            max_tokens: Approximate token limit
            
        Returns:
            List of recent messages
        """
        # Simple approach: take last N messages
        # In production, use proper token counting
        recent = self.messages[-max_messages:]
        
        # Approximate token count (4 chars per token)
        total_chars = sum(len(m.content) for m in recent)
        estimated_tokens = total_chars // 4
        
        # Trim if exceeds token limit
        while estimated_tokens > max_tokens and len(recent) > 1:
            recent = recent[1:]  # Remove oldest
            total_chars = sum(len(m.content) for m in recent)
            estimated_tokens = total_chars // 4
        
        return recent
    
    def to_langchain_messages(self) -> list[tuple[str, str]]:
        """Convert to LangChain message format."""
        return [(m.role, m.content) for m in self.messages]


class ConversationSummary(BaseModel):
    """Summary of a conversation for listing."""
    
    id: str
    title: Optional[str]
    document_id: Optional[str]
    message_count: int
    created_at: datetime
    updated_at: datetime
    preview: str  # First few words of last message

