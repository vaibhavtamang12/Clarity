"""Generation evaluation metrics (Phase 22).

Metrics distinct from retrieval metrics:
- Faithfulness: Are claims in the answer supported by retrieved context?
- Answer correctness: Does the answer match the reference answer?
- Answer relevance: Does the answer address the question?
- Citation correctness: Do citations actually support the claims they're cited for?
- Context relevance: Is the retrieved context relevant to the question?

LLM-as-judge is used for semantic metrics (faithfulness, answer correctness,
answer relevance). This is a production pattern, not a toy — the judge is
the same LLM abstraction from Phase 10, so it's configurable and swappable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.base import ChatMessage, LLMProvider, LLMRequest

logger = get_logger(__name__)


@dataclass(frozen=True)
class GenerationMetrics:
    """Generation evaluation metrics for a single question."""

    question_id: str
    category: str
    configuration: str
    faithfulness: float
    answer_correctness: float
    answer_relevance: float
    citation_correctness: float
    context_relevance: float
    answer_text: str
    reference_answer: str
    num_citations: int
    num_claims: int
    unsupported_claims: int


# ------------------------------------------------------------------ LLM judges
JUDGE_SYSTEM_PROMPT = """You are an impartial evaluation judge for a RAG system.
Your task is to evaluate the quality of a generated answer.

Rules:
- Evaluate ONLY based on the provided context and question.
- Do NOT use outside knowledge.
- Output STRICT JSON matching the requested schema.
- Scores are floats from 0.0 (poor) to 1.0 (excellent).
"""


def _parse_judge_response(content: str) -> dict:
    """Parse JSON from LLM judge response."""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in judge response")
    return json.loads(text[start:end])


async def judge_faithfulness(
    provider: LLMProvider,
    question: str,
    answer: str,
    context: str,
) -> float:
    """Judge whether the answer is faithful to the retrieved context.

    Faithfulness = are the claims in the answer supported by the context?
    Returns 0.0-1.0.
    """
    prompt = f"""Question: {question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Evaluate the FAITHFULNESS of the answer:
- Does the answer only use information from the retrieved context?
- Are there any claims in the answer NOT supported by the context?
- Does the answer fabricate information?

Output STRICT JSON: {{"faithfulness": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    try:
        response = await provider.generate(
            LLMRequest(
                messages=[
                    ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=200,
                json_output=True,
            )
        )
        result = _parse_judge_response(response.content)
        return float(result.get("faithfulness", 0.0))
    except Exception as exc:
        logger.warning("faithfulness_judge_failed", error=type(exc).__name__)
        return 0.0


async def judge_answer_correctness(
    provider: LLMProvider,
    question: str,
    answer: str,
    reference_answer: str,
) -> float:
    """Judge whether the answer is correct compared to the reference.

    Answer correctness = does the answer match the reference answer?
    Returns 0.0-1.0.
    """
    prompt = f"""Question: {question}

Reference Answer (ground truth):
{reference_answer}

Generated Answer:
{answer}

Evaluate the CORRECTNESS of the generated answer:
- Does it match the factual content of the reference answer?
- Are there any factual errors or contradictions?
- Is it complete and accurate?

Output STRICT JSON: {{"correctness": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    try:
        response = await provider.generate(
            LLMRequest(
                messages=[
                    ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=200,
                json_output=True,
            )
        )
        result = _parse_judge_response(response.content)
        return float(result.get("correctness", 0.0))
    except Exception as exc:
        logger.warning("correctness_judge_failed", error=type(exc).__name__)
        return 0.0


async def judge_answer_relevance(
    provider: LLMProvider,
    question: str,
    answer: str,
) -> float:
    """Judge whether the answer is relevant to the question.

    Answer relevance = does the answer address what was asked?
    Returns 0.0-1.0.
    """
    prompt = f"""Question: {question}

Generated Answer:
{answer}

Evaluate the RELEVANCE of the answer:
- Does it directly address the question asked?
- Does it contain irrelevant information or tangents?
- Is it focused and on-topic?

Output STRICT JSON: {{"relevance": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    try:
        response = await provider.generate(
            LLMRequest(
                messages=[
                    ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=200,
                json_output=True,
            )
        )
        result = _parse_judge_response(response.content)
        return float(result.get("relevance", 0.0))
    except Exception as exc:
        logger.warning("relevance_judge_failed", error=type(exc).__name__)
        return 0.0


async def judge_context_relevance(
    provider: LLMProvider,
    question: str,
    context: str,
) -> float:
    """Judge whether the retrieved context is relevant to the question.

    Context relevance = is the retrieved context useful for answering?
    Returns 0.0-1.0.
    """
    prompt = f"""Question: {question}

Retrieved Context:
{context}

Evaluate the RELEVANCE of the retrieved context:
- Is this context relevant to answering the question?
- Does it contain the information needed to answer?
- Is it focused, or does it include irrelevant documents?

Output STRICT JSON: {{"context_relevance": <float 0.0-1.0>, "explanation": "<one sentence>"}}"""

    try:
        response = await provider.generate(
            LLMRequest(
                messages=[
                    ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.0,
                max_tokens=200,
                json_output=True,
            )
        )
        result = _parse_judge_response(response.content)
        return float(result.get("context_relevance", 0.0))
    except Exception as exc:
        logger.warning("context_relevance_judge_failed", error=type(exc).__name__)
        return 0.0


# ------------------------------------------------------------------ citation correctness
def compute_citation_correctness(
    citations_with_claims: Sequence[tuple[str, str]],
    chunk_contents: dict[str, str],
) -> float:
    """Compute citation correctness using Phase 12's validation logic.

    citations_with_claims: list of (chunk_id, claim) tuples
    chunk_contents: mapping from chunk_id to chunk content
    Returns 0.0-1.0.
    """
    from app.evaluation.relevance import is_relevant

    if not citations_with_claims:
        return 1.0  # no citations = nothing to be wrong about

    correct = 0
    for chunk_id, claim in citations_with_claims:
        chunk_content = chunk_contents.get(chunk_id, "")
        if not chunk_content:
            continue
        if is_relevant(chunk_content, claim):
            correct += 1

    return correct / len(citations_with_claims) if citations_with_claims else 1.0