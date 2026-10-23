# DECISIONS

Format: date — decision. Options · evidence · choice · how to reverse.

## 2026-10-23 — CLAUDE.md imports kickoff + PLAN, not the whole PROGRESS.md
Options: import PROGRESS.md (plan §11) vs read its tail. PROGRESS.md is append-only and will grow to thousands of lines; importing it would burn context every turn. Choice: CLAUDE.md imports docs/kickoff.md and PLAN.md; rule 1 + the SessionStart compact hook bring in `tail -40 PROGRESS.md`. Reverse: add `@PROGRESS.md` to CLAUDE.md.
