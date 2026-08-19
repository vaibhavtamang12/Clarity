"""Security package (Phase 25).

Deliberately outside the layered packages: security concerns cut across all
layers (ingestion, generation, API), and the import-linter layer contract
does not constrain this module — the same treatment as app.container.
"""