# Academic Research Skills (ARS) v3.6.7

This project has ARS skills available via the global skills directory (`~/.Codex/skills/`). These skills assist with academic research, paper writing, peer review, and pipeline orchestration.

## Available Skills

### `/deep-research` (v2.9.3)
13-agent research team for deep investigation. Modes: `full`, `quick`, `socratic`, `review`, `lit-review`, `fact-check`, `systematic-review`.

### `/academic-paper` (v3.1.1)
12-agent paper writing team. Modes: `full`, `plan`, `outline-only`, `revision`, `revision-coach`, `abstract-only`, `lit-review`, `format-convert`, `citation-check`, `disclosure`.

### `/academic-paper-reviewer` (v1.9.0)
Multi-perspective peer review simulator. Modes: `full`, `re-review`, `quick`, `methodology-focus`, `guided`, `calibration`.

### `/academic-pipeline` (v3.6.7)
Full pipeline orchestrator coordinating all above skills. 10-stage workflow.

## Routing Rules

- Use individual skills directly when only one function is needed (e.g., `/deep-research` for research, `/academic-paper` for writing).
- Use `/academic-pipeline` for end-to-end workflow from research to publication.
- `deep-research` is the upstream research engine; `academic-paper` is the downstream publication engine.
- Socratic/plan modes = guided dialogue; full modes = direct production.
- Guided mode for review = Socratic engagement to learn; full = standard report.

## Key Rules

- All claims must have citations with evidence hierarchy respected.
- Contradictions must be disclosed transparently.
- AI disclosure required in all reports.
- Default output language matches user input language.

## Pipeline Flow

`deep-research` → `academic-paper` → Integrity Check → `academic-paper-reviewer` → Revision → Re-review (max 2 loops) → Final Integrity Check → Format Convert → Output + Process Summary

## Handoff Protocols

- deep-research → academic-paper: RQ Brief, Methodology Blueprint, Annotated Bibliography, Synthesis Report, INSIGHT Collection
- academic-paper → academic-paper-reviewer: Complete paper text with domain auto-detection
- academic-paper-reviewer → academic-paper: Editorial Decision Letter, Revision Roadmap, per-reviewer comments


<claude-mem-context>
# Memory Context

# [论文-1.1] recent context, 2026-05-11 1:34pm GMT+8

No previous sessions found.
</claude-mem-context>