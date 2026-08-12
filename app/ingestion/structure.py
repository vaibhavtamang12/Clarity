"""Structure annotation: converts heading blocks into breadcrumbs.

After this pass, every block knows the heading path it lives under, e.g.
("Policy", "Refunds", "Enterprise"). Chunkers copy that path into chunk
metadata — this is what makes section-level citations possible.
"""

from __future__ import annotations

from app.ingestion.domain import ParserOutput, StructuralBlock


def annotate_sections(output: ParserOutput) -> ParserOutput:
    current_headings: dict[int, str] = {}

    def path() -> tuple[str, ...]:
        return tuple(current_headings[level] for level in sorted(current_headings))

    annotated: list[StructuralBlock] = []
    for block in output.blocks:
        if block.is_heading:
            level = max(1, block.heading_level)
            current_headings[level] = block.text.strip()
            # A new heading at level N invalidates all deeper headings.
            for deeper in [lvl for lvl in current_headings if lvl > level]:
                del current_headings[deeper]
            block.section_path = path()
        else:
            block.section_path = path()
        annotated.append(block)

    return ParserOutput(blocks=annotated, metadata=output.metadata)