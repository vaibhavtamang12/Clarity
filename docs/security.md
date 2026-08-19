# Security Model — RAG Knowledge Intelligence Platform (Phase 25)

## Doctrine

**A document is data, never instructions.** Every defense in this system
derives from that single sentence.

## Threat Model

| Threat | Vector | Defense | Status |
|--------|--------|---------|--------|
| Prompt injection via document content | Malicious text inside ingested documents reaches the LLM prompt | Structural: XML escaping of all passage content + attributes; behavioral: hardened prompt with anti-manipulation rules; detective: pattern scanner → security events | Mitigated (structural); model-level red-team = future work |
| Passage breakout | Document contains `</passage>` or fake tags to forge prompt structure | `escape_passage_content()` removes the escape mechanism entirely | Mitigated |
| System prompt exfiltration | Document instructs model to reveal its instructions | Prompt rules + escaping; judge/output validation never echo prompts | Mitigated |
| SSRF via URL ingestion | `POST /documents/url` pointed at internal/metadata endpoints | Scheme allow-list, credential rejection, DNS resolution, private/link-local/reserved IP blocking, per-hop redirect revalidation | Mitigated; DNS-rebinding residual documented below |
| Memory exhaustion via URL | URL serves an enormous body | Streaming fetch with incremental size cap | Mitigated |
| Zip bomb / zip slip via DOCX | Crafted archive explodes parser memory or writes paths | Entry-count cap, uncompressed-size cap, unsafe-path rejection at validation time | Mitigated |
| Parser DoS via PDF | Thousands of pages | Page-count cap at validation time | Mitigated |
| Tenant data leakage | User A accesses User B's documents/conversations/search results | Ownership checks return 404 (existence never leaked); search filters owner_id at the retrieval layer; response cache keys are per-user | Mitigated |
| Stale cache after version change | Old answers served after document updates | Cache keys embed the index-state counter — keys change, nothing is deleted | Mitigated by construction |
| Secret leakage | API keys/LLM keys in logs or error responses | `SecretStr` settings, hashed key storage, `sanitize_fields` redaction, generic error envelopes | Mitigated |
| Rate-limit abuse | Brute force / cost attacks | Sliding-window rate limiting per user (Redis) | Mitigated |
| Metrics scrape exposure | `/metrics` unauthenticated by Prometheus convention | Network-level protection in production deployments (documented) | Accepted risk with documented control |

## Defense Layers for Prompt Injection

1. **Structural isolation (primary):** all passage content and attributes are
   XML-escaped before entering the prompt. A payload cannot close a tag, open
   a tag, or impersonate prompt structure — regardless of what it says.
2. **Behavioral (secondary):** the system prompt declares passages untrusted
   and forbids role changes, instruction-following from content, and prompt
   disclosure. Prompt version: `rag-grounded/2`.
3. **Detective (tertiary):** known injection signatures are scanned and
   recorded as `security_events_total{event_type="injection_pattern_detected"}`.
   Detection ALERTS; it never blocks — blocking on content patterns is a
   denial-of-service vector (D-138).

## Residual Risks (honest, per Rule 8)

- **DNS rebinding:** validation resolves DNS, then httpx connects separately;
  a rebinding attack in that window is theoretically possible. Full mitigation
  requires an IP-pinned transport (future work, tracked for Phase 29+).
- **Model-level manipulation:** novel payloads that convince the model despite
  escaping/framing require red-team testing with a real LLM — structural
  defenses bound the blast radius, but behavioral testing is future work.
- **Metrics endpoint:** unauthenticated by Prometheus convention; production
  deployments must place it behind network policy.

## Security Event Types

| Event | Meaning |
|-------|---------|
| `injection_pattern_detected` | Known injection signature found in retrieved content |
| `ssrf_blocked` | URL ingestion attempt rejected by SSRF policy |
| `auth_failure` | Invalid/expired API key rejected |
| `malicious_file_rejected` | File failed binary safety validation |
| `ownership_violation` | Cross-tenant access attempt (404 returned) |

All events are counted in Prometheus and logged (sanitized) for audit.