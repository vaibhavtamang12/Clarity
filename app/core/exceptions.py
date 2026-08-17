"""Domain exception hierarchy with stable, machine-readable error codes.

Maps directly to the API error envelope in docs/ARCHITECTURE.md section 6.
Callers raise typed exceptions; the API layer converts them to the envelope.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    RATE_LIMITED = "RATE_LIMITED"
    NOT_FOUND = "NOT_FOUND"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    CONVERSATION_NOT_FOUND = "CONVERSATION_NOT_FOUND"
    CONFLICT = "CONFLICT"
    UNSUPPORTED_DOCUMENT_TYPE = "UNSUPPORTED_DOCUMENT_TYPE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INGESTION_FAILED = "INGESTION_FAILED"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    EMBEDDING_UNAVAILABLE = "EMBEDDING_UNAVAILABLE"
    HTTP_ERROR = "HTTP_ERROR"
    INTERNAL = "INTERNAL"
    INGESTION_VALIDATION = "INGESTION_VALIDATION"



class AppError(Exception):
    """Base class for all domain errors."""

    code: ErrorCode = ErrorCode.INTERNAL
    status_code: int = 500

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        status_code: int | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        self.details = details or {}


class ConfigurationError(AppError):
    code = ErrorCode.CONFIGURATION_ERROR
    status_code = 500


class NotFoundError(AppError):
    code = ErrorCode.NOT_FOUND
    status_code = 404


class DocumentNotFoundError(NotFoundError):
    code = ErrorCode.DOCUMENT_NOT_FOUND


class ConversationNotFoundError(NotFoundError):
    code = ErrorCode.CONVERSATION_NOT_FOUND


class ConflictError(AppError):
    code = ErrorCode.CONFLICT
    status_code = 409


class UnauthorizedError(AppError):
    code = ErrorCode.UNAUTHORIZED
    status_code = 401


class RateLimitedError(AppError):
    code = ErrorCode.RATE_LIMITED
    status_code = 429


class UnsupportedDocumentTypeError(AppError):
    code = ErrorCode.UNSUPPORTED_DOCUMENT_TYPE
    status_code = 400


class IngestionError(AppError):
    code = ErrorCode.INGESTION_FAILED
    status_code = 500


class ExternalServiceError(AppError):
    status_code = 502


class RetrievalUnavailableError(ExternalServiceError):
    code = ErrorCode.RETRIEVAL_UNAVAILABLE
    status_code = 503


class LLMUnavailableError(ExternalServiceError):
    code = ErrorCode.LLM_UNAVAILABLE
    status_code = 503


class EmbeddingUnavailableError(ExternalServiceError):
    code = ErrorCode.EMBEDDING_UNAVAILABLE
    status_code = 503

class IngestionValidationError(AppError):
    """Input rejected before any processing (bad type, size, or URL policy)."""

    code = ErrorCode.INGESTION_VALIDATION
    status_code = 400

# Add class (code RETRIEVAL_UNAVAILABLE already exists):
class VectorStoreUnavailableError(ExternalServiceError):
    """Qdrant (or any vector store) failed after retries."""

    code = ErrorCode.RETRIEVAL_UNAVAILABLE
    status_code = 503

# Add class:
class RerankerUnavailableError(ExternalServiceError):
    """Reranker model failed to load or score. Retrieval itself still succeeded,
    so callers may choose to degrade instead of failing the request (D-049)."""

    code = ErrorCode.RETRIEVAL_UNAVAILABLE
    status_code = 503