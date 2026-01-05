"""PostgreSQL-based conversation storage for multi-turn context.

Provides persistent storage for conversation history, enabling
multi-turn context in RAG interactions.
"""

import json
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Boolean,
    Integer,
    ForeignKey,
    select,
    update,
    delete,
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase, relationship

from app.config import settings
from app.memory.models import (
    Conversation,
    Message,
    MessageRole,
    ConversationSummary,
)

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


class ConversationRecord(Base):
    """Database model for conversations."""
    
    __tablename__ = "conversations"
    
    id = Column(String(36), primary_key=True)
    user_id = Column(String(255), nullable=True, index=True)
    document_id = Column(String(255), nullable=True, index=True)
    title = Column(String(500), nullable=True)
    summary = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    messages = relationship(
        "MessageRecord",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageRecord.timestamp",
    )


class MessageRecord(Base):
    """Database model for messages."""
    
    __tablename__ = "messages"
    
    id = Column(String(36), primary_key=True)
    conversation_id = Column(
        String(36),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    sources_json = Column(Text, nullable=True)
    tokens_used = Column(Integer, nullable=True)
    latency_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("ConversationRecord", back_populates="messages")


class ConversationStore:
    """
    PostgreSQL-based store for conversation history.
    
    Provides CRUD operations for conversations and messages,
    supporting multi-turn context in RAG interactions.
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize the store.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        # Convert to async URL
        url = database_url or settings.postgres_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        self.engine = create_async_engine(
            url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
        )
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Conversation store initialized")

    @asynccontextmanager
    async def session(self):
        """Get an async session context manager."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def create_conversation(
        self,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            user_id: Optional user identifier
            document_id: Optional document scope
            title: Optional conversation title
            
        Returns:
            New Conversation object
        """
        conversation = Conversation(
            user_id=user_id,
            document_id=document_id,
            title=title,
        )
        
        async with self.session() as session:
            record = ConversationRecord(
                id=conversation.id,
                user_id=conversation.user_id,
                document_id=conversation.document_id,
                title=conversation.title,
                is_active=True,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )
            session.add(record)
        
        logger.debug(f"Created conversation: {conversation.id}")
        return conversation

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> Optional[Conversation]:
        """
        Get a conversation by ID with all messages.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Conversation or None if not found
        """
        async with self.session() as session:
            result = await session.execute(
                select(ConversationRecord).where(
                    ConversationRecord.id == conversation_id
                )
            )
            record = result.scalar_one_or_none()
            
            if not record:
                return None
            
            # Load messages
            msg_result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.timestamp)
            )
            message_records = msg_result.scalars().all()
            
            messages = [
                Message(
                    id=m.id,
                    role=MessageRole(m.role),
                    content=m.content,
                    timestamp=m.timestamp,
                    sources=json.loads(m.sources_json) if m.sources_json else None,
                    tokens_used=m.tokens_used,
                    latency_ms=m.latency_ms,
                )
                for m in message_records
            ]
            
            return Conversation(
                id=record.id,
                user_id=record.user_id,
                document_id=record.document_id,
                title=record.title,
                messages=messages,
                created_at=record.created_at,
                updated_at=record.updated_at,
                is_active=record.is_active,
                summary=record.summary,
            )

    async def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        sources: Optional[list[dict]] = None,
        tokens_used: Optional[int] = None,
        latency_ms: Optional[int] = None,
    ) -> Message:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Target conversation
            role: Message role
            content: Message content
            sources: Retrieved sources
            tokens_used: Tokens used for response
            latency_ms: Response latency
            
        Returns:
            Created Message
        """
        message = Message(
            role=role,
            content=content,
            sources=sources,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
        )
        
        async with self.session() as session:
            record = MessageRecord(
                id=message.id,
                conversation_id=conversation_id,
                role=role.value,
                content=content,
                sources_json=json.dumps(sources) if sources else None,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                timestamp=message.timestamp,
            )
            session.add(record)
            
            # Update conversation timestamp
            await session.execute(
                update(ConversationRecord)
                .where(ConversationRecord.id == conversation_id)
                .values(updated_at=datetime.utcnow())
            )
        
        return message

    async def list_conversations(
        self,
        user_id: Optional[str] = None,
        document_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ConversationSummary]:
        """
        List conversations with summaries.
        
        Args:
            user_id: Filter by user
            document_id: Filter by document
            limit: Max results
            offset: Pagination offset
            
        Returns:
            List of conversation summaries
        """
        async with self.session() as session:
            query = select(ConversationRecord).where(
                ConversationRecord.is_active == True  # noqa: E712
            )
            
            if user_id:
                query = query.where(ConversationRecord.user_id == user_id)
            if document_id:
                query = query.where(ConversationRecord.document_id == document_id)
            
            query = query.order_by(ConversationRecord.updated_at.desc())
            query = query.limit(limit).offset(offset)
            
            result = await session.execute(query)
            records = result.scalars().all()
            
            summaries = []
            for record in records:
                # Get last message for preview
                msg_result = await session.execute(
                    select(MessageRecord)
                    .where(MessageRecord.conversation_id == record.id)
                    .order_by(MessageRecord.timestamp.desc())
                    .limit(1)
                )
                last_msg = msg_result.scalar_one_or_none()
                preview = (
                    last_msg.content[:100] + "..."
                    if last_msg and len(last_msg.content) > 100
                    else (last_msg.content if last_msg else "")
                )
                
                # Get message count
                count_result = await session.execute(
                    select(MessageRecord)
                    .where(MessageRecord.conversation_id == record.id)
                )
                message_count = len(count_result.scalars().all())
                
                summaries.append(ConversationSummary(
                    id=record.id,
                    title=record.title,
                    document_id=record.document_id,
                    message_count=message_count,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    preview=preview,
                ))
            
            return summaries

    async def update_summary(
        self,
        conversation_id: str,
        summary: str,
    ) -> None:
        """
        Update conversation summary.
        
        Args:
            conversation_id: Conversation ID
            summary: Generated summary
        """
        async with self.session() as session:
            await session.execute(
                update(ConversationRecord)
                .where(ConversationRecord.id == conversation_id)
                .values(summary=summary)
            )

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> bool:
        """
        Delete a conversation and all its messages.
        
        Args:
            conversation_id: Conversation to delete
            
        Returns:
            True if deleted
        """
        async with self.session() as session:
            result = await session.execute(
                delete(ConversationRecord)
                .where(ConversationRecord.id == conversation_id)
            )
            return result.rowcount > 0

    async def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: int = 10,
    ) -> list[tuple[str, str]]:
        """
        Get conversation context for LLM.
        
        Returns messages in (role, content) format suitable
        for LangChain chat prompts.
        
        Args:
            conversation_id: Conversation ID
            max_messages: Maximum messages to return
            
        Returns:
            List of (role, content) tuples
        """
        async with self.session() as session:
            result = await session.execute(
                select(MessageRecord)
                .where(MessageRecord.conversation_id == conversation_id)
                .order_by(MessageRecord.timestamp.desc())
                .limit(max_messages)
            )
            records = list(reversed(result.scalars().all()))
            
            return [(m.role, m.content) for m in records]


# Singleton instance
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get or create conversation store singleton."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store

