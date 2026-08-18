# app/api/v1/endpoints/search.py
"""Direct retrieval endpoint (no generation) with tenant isolation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.api.auth import AuthenticatedUser
from app.api.deps import get_current_user, get_db, get_platform
from app.repositories.vector.base import VectorFilter
from app.schemas.search import SearchHit, SearchRequest, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    request: Request,
    session=Depends(get_db),
    user: AuthenticatedUser = Depends(get_current_user),
) -> SearchResponse:
    platform = get_platform(request)
    # owner_id is enforced in the filter — tenant isolation at the retrieval layer.
    vf = VectorFilter(
        owner_id=user.id,
        document_id=payload.document_id,
        version_id=payload.version_id,
    )
    result = await platform.retriever.retrieve(payload.query, top_k=payload.top_k, filter_=vf)
    return SearchResponse(
        items=[
            SearchHit(
                chunk_id=item.chunk_id, document_id=item.document_id, version_id=item.version_id,
                score=item.score, sources=list(item.sources),
                dense_score=item.dense_score, sparse_score=item.sparse_score,
                rerank_score=item.rerank_score, content=item.content,
                section=item.section, source_uri=item.source_uri,
                page_start=item.page_start, page_end=item.page_end,
            )
            for item in result.items
        ],
        retriever=result.metadata.retriever,
        strategy=result.metadata.strategy,
        degraded=result.metadata.degraded,
        branch_latencies_ms=result.metadata.branch_latencies_ms,
    )