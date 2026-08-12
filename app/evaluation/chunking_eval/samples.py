"""Gold evaluation samples for chunking experiments (30 questions).

Each sample carries verbatim evidence spans copied from the corpus — the
strategy-agnostic ground truth described in app/evaluation/relevance.py.
Coverage: simple factual lookups across all four documents, section-specific
facts, and number-heavy facts that stress precise chunk boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SAMPLES_VERSION = "chunking-samples-v1"


@dataclass(frozen=True)
class ChunkingEvalSample:
    sample_id: str
    question: str
    evidence: tuple[str, ...]
    source_document: str
    tags: tuple[str, ...] = field(default_factory=tuple)


def build_chunking_samples() -> list[ChunkingEvalSample]:
    return [
        # ------------------------------------------------ refund_policy.md
        ChunkingEvalSample(
            "refund-01",
            "How long is the refund window for enterprise customers?",
            ("Enterprise customers have a 30-day refund window from the invoice date.",),
            "refund_policy.md",
            ("factual", "enterprise"),
        ),
        ChunkingEvalSample(
            "refund-02",
            "Which email address must receive the written refund notice?",
            ("Written notice must be sent to refunds@company.com.",),
            "refund_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "refund-03",
            "How long does the finance team take to approve enterprise refunds?",
            ("The finance team approves enterprise refunds within 5 business days.",),
            "refund_policy.md",
            ("factual", "enterprise"),
        ),
        ChunkingEvalSample(
            "refund-04",
            "Which purchases are excluded from refunds entirely?",
            (
                "Custom negotiated contracts are excluded from this policy.",
                "Professional services fees are non-refundable once work has started.",
                "Domain purchases cannot be refunded after registration.",
            ),
            "refund_policy.md",
            ("multi-evidence",),
        ),
        ChunkingEvalSample(
            "refund-05",
            "How are annual contracts prorated when refunded?",
            ("Annual contracts are prorated by full months of unused service.",),
            "refund_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "refund-06",
            "How long does an approved refund take to reach the original payment method?",
            ("Approved refunds are returned to the original payment method within 10 business days.",),
            "refund_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "refund-07",
            "How quickly does support validate a refund request?",
            ("Support validates the request within 2 business days.",),
            "refund_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "refund-08",
            "What is the standard refund window for regular plans?",
            ("The standard refund window for regular plans is 14 days from the purchase date.",),
            "refund_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "refund-09",
            "What must every refund request start with?",
            ("All refunds require a support ticket opened by the account owner.",),
            "refund_policy.md",
            ("factual",),
        ),
        # ---------------------------------------------- security_policy.md
        ChunkingEvalSample(
            "security-01",
            "How often are access reviews performed?",
            ("Access reviews are performed quarterly by system owners.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-02",
            "What is the minimum password length?",
            ("Passwords must be at least 14 characters long.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-03",
            "Is multi-factor authentication optional for employees?",
            ("Multi-factor authentication is mandatory for all accounts.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-04",
            "After how many minutes of inactivity does the screen lock activate?",
            ("Screen lock activates after 5 minutes of inactivity.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-05",
            "What is the reporting deadline for a security incident?",
            ("Security incidents must be reported to security@company.com within 1 hour of discovery.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-06",
            "How soon after resolving a major incident is a postmortem required?",
            ("A postmortem is required within 5 business days of resolving a P1 or P2 incident.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-07",
            "How long are system logs retained?",
            ("System logs are retained for 365 days.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-08",
            "Within what period must data deletion requests be completed?",
            ("Verified data deletion requests are completed within 30 days.",),
            "security_policy.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "security-09",
            "Are USB storage devices allowed on company hardware?",
            ("USB storage devices are prohibited on company hardware.",),
            "security_policy.md",
            ("factual",),
        ),
        # ------------------------------------------------- api_reference.md
        ChunkingEvalSample(
            "api-01",
            "How do API requests authenticate?",
            ("All requests authenticate with bearer tokens sent in the Authorization header.",),
            "api_reference.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "api-02",
            "How long does an API token remain valid?",
            ("Tokens expire after 3600 seconds and must be refreshed.",),
            "api_reference.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "api-03",
            "What is the per-key rate limit of the API?",
            ("Each API key is limited to 60 requests per minute.",),
            "api_reference.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "api-04",
            "Which HTTP status code is returned when rate limited?",
            ("Rate-limited requests receive HTTP status 429 with a Retry-After header.",),
            "api_reference.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "api-05",
            "What is the maximum page size for list endpoints?",
            ("The maximum allowed limit is 100 items per page.",),
            "api_reference.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "api-06",
            "How much notice is given before a deprecated endpoint is removed?",
            ("Deprecated endpoints remain available for 6 months after the deprecation notice.",),
            "api_reference.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "api-07",
            "What prefix do API keys use?",
            ("API keys use the rk_ prefix.",),
            "api_reference.md",
            ("factual",),
        ),
        # ---------------------------------------------- onboarding_guide.md
        ChunkingEvalSample(
            "onboarding-01",
            "When is a buddy assigned to a new hire?",
            ("Every new hire is assigned a buddy on day one.",),
            "onboarding_guide.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "onboarding-02",
            "How much is the home office monitor stipend?",
            ("A monitor stipend of 300 dollars is provided for home office setup.",),
            "onboarding_guide.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "onboarding-03",
            "What is the deadline for completing security training?",
            ("Security training must be completed within 14 days of joining.",),
            "onboarding_guide.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "onboarding-04",
            "When does the probation review take place?",
            ("The probation review happens at 90 days.",),
            "onboarding_guide.md",
            ("factual",),
        ),
        ChunkingEvalSample(
            "onboarding-05",
            "What is required before production system access is granted?",
            ("Production system access requires completing access training first.",),
            "onboarding_guide.md",
            ("factual",),
        ),
    ]