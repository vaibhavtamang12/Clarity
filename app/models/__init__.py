"""SQLAlchemy ORM models (docs/ARCHITECTURE.md §5.1).

Importing this package registers every model on Base.metadata — this is
what Alembic's target_metadata and the integration fixtures rely on.
"""

from app.models.base import Base
from app.models.conversation import Conversation, Message
from app.models.document import Document, DocumentChunk, DocumentVersion
from app.models.job import IngestionJob
from app.models.logs import EvaluationRun, RetrievalLog
from app.models.user import ApiKey, User

__all__ = [
    "ApiKey",
    "Base",
    "Conversation",
    "Document",
    "DocumentChunk",
    "DocumentVersion",
    "EvaluationRun",
    "IngestionJob",
    "Message",
    "RetrievalLog",
    "User",
]