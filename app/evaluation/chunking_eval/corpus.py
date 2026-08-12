"""Chunking experiment corpus — realistic, deterministic, code-generated.

Four Markdown documents with distinct topics and section structures.
Facts are specific and single-sourced so each evaluation question has
exactly one evidence span. The corpus runs through the REAL ingestion
path (MarkdownParser → cleaner → structure annotator → chunker): no
experiment-only code paths (Rule 15).
"""

from __future__ import annotations

from dataclasses import dataclass

CORPUS_VERSION = "chunking-corpus-v1"


@dataclass(frozen=True)
class CorpusDocument:
    name: str
    text: str


REFUND_POLICY = """# Refund Policy

This policy describes when and how refunds are issued for all subscription plans.

## Overview

All refunds require a support ticket opened by the account owner. The standard refund window for regular plans is 14 days from the purchase date. Refund requests outside the window are reviewed case by case.

## Enterprise Refunds

Enterprise customers have a 30-day refund window from the invoice date. The subscription must be unused at the time of the request. Written notice is required for every enterprise refund. The finance team approves enterprise refunds within 5 business days.

## Conditions

The subscription must be unused at the time of the request. Written notice must be sent to refunds@company.com. Annual contracts are prorated by full months of unused service. Taxes are refunded according to local regulations.

## Exclusions

Custom negotiated contracts are excluded from this policy. Professional services fees are non-refundable once work has started. Domain purchases cannot be refunded after registration.

## Refund Process

The customer opens a support ticket requesting a refund. Support validates the request within 2 business days. Finance approves valid requests within 5 business days. Approved refunds are returned to the original payment method within 10 business days.
"""

SECURITY_POLICY = """# Security Policy

This policy defines mandatory security controls for all employees and contractors.

## Access Control

Single sign-on is required for all internal systems. Access reviews are performed quarterly by system owners. Access is granted on a least-privilege basis and revoked within 24 hours of role change.

## Password Requirements

Passwords must be at least 14 characters long. Multi-factor authentication is mandatory for all accounts. Approved password managers must be used for storing credentials. Password reuse across systems is prohibited.

## Device Policy

Full disk encryption is mandatory on all company devices. Screen lock activates after 5 minutes of inactivity. USB storage devices are prohibited on company hardware. Lost devices must be reported within 4 hours.

## Incident Response

Security incidents must be reported to security@company.com within 1 hour of discovery. Incidents are classified by severity from P1 to P4. A postmortem is required within 5 business days of resolving a P1 or P2 incident.

## Data Retention

System logs are retained for 365 days. Backups are retained for 90 days. Verified data deletion requests are completed within 30 days.
"""

API_REFERENCE = """# API Reference

This reference documents the public REST API for the knowledge platform.

## Authentication

All requests authenticate with bearer tokens sent in the Authorization header. API keys use the rk_ prefix. Tokens expire after 3600 seconds and must be refreshed.

## Rate Limits

Each API key is limited to 60 requests per minute. Bursts of up to 10 concurrent requests are allowed. Rate-limited requests receive HTTP status 429 with a Retry-After header.

## Error Codes

Status 400 indicates a malformed request. Status 401 indicates missing or invalid authentication. Status 404 indicates the resource does not exist. Status 422 indicates request validation failure. Status 500 indicates an internal server error.

## Pagination

List endpoints accept limit and offset parameters. The maximum allowed limit is 100 items per page. Responses include total_count for paging through results.

## Versioning

The API is versioned in the URL path under /api/v1. Breaking changes are only introduced in new major versions. Deprecated endpoints remain available for 6 months after the deprecation notice.
"""

ONBOARDING_GUIDE = """# Onboarding Guide

This guide covers everything a new hire needs during the first 90 days.

## First Week

Every new hire is assigned a buddy on day one. HR orientation happens on day 3 of the first week. The manager schedules a kickoff meeting within the first two days.

## Equipment

New hires choose between a MacBook or a ThinkPad. A monitor stipend of 300 dollars is provided for home office setup. Equipment is shipped before the start date.

## Accounts and Access

Single sign-on access is provisioned on day one. Production system access requires completing access training first. All access requests are audited monthly.

## Training

Security training must be completed within 14 days of joining. Role-specific training is completed by day 30. Training progress is tracked in the learning portal.

## Probation Review

The probation review happens at 90 days. The review includes meetings with the manager and the skip-level manager. Probation outcomes are documented in the HR system.
"""


def build_chunking_corpus() -> list[CorpusDocument]:
    return [
        CorpusDocument(name="refund_policy.md", text=REFUND_POLICY),
        CorpusDocument(name="security_policy.md", text=SECURITY_POLICY),
        CorpusDocument(name="api_reference.md", text=API_REFERENCE),
        CorpusDocument(name="onboarding_guide.md", text=ONBOARDING_GUIDE),
    ]