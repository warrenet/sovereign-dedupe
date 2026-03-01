# AGENTS.md — Sovereign Dedupe
*Deploy to: project root of Sovereign Dedupe repo*
*Last updated: S81 | 2026-03-01*

---

## What This Is

Sovereign Dedupe is Warren's file deduplication system — a single-file Python script (stdlib-only, zero external deps) that audits, stages, and rolls back duplicate files using SHA-256 verification. DRY-RUN by default; destructive actions require explicit `--apply --confirm <PLAN_ID>`. v1.7.2-bluewin.

---

## System Architecture

```
sovereign_dedupe.py (single-file, stdlib-only)
    ↓
audit → generates dedup plan (PLAN_ID + plan_sha256)
    ↓
execute → stages duplicates (reversible, not deleted)
    ↓
rollback → restores from manifest
```

**Staging directory:** `Sovereign_Staging/` — files moved here, never permanently deleted until explicit cleanup.

---

## Stack

- **Language:** Python 3 (stdlib-only — no pip install required)
- **Runner:** `run.ps1` (Windows PowerShell)
- **Key files:** `sovereign_dedupe.py`, `run.ps1`

---

## Command Reference

```bash
# Audit (dry-run, safe)
python sovereign_dedupe.py audit --preview 10

# Execute (stages duplicates, DRY-RUN by default)
python sovereign_dedupe.py execute --verify-hash --max-moves 25

# Execute with apply (DESTRUCTIVE — moves files to staging)
python sovereign_dedupe.py execute --verify-hash --max-moves 25 --apply --confirm <PLAN_ID>

# Rollback (restores from staging manifest)
python sovereign_dedupe.py rollback --manifest Sovereign_Staging/sovereign_manifest_*.json --apply

# Selftest (10+ tests, safe)
python sovereign_dedupe.py selftest
```

**Windows shorthand via run.ps1:**
```powershell
.\run.ps1 -cmd audit
.\run.ps1 -cmd selftest
.\run.ps1 -cmd execute -Apply -Confirm <PLAN_ID>
.\run.ps1 -cmd rollback -Apply
```

---

## Rules

**Before declaring done:**
1. Run `python sovereign_dedupe.py selftest` — all 10+ tests pass
2. Run `python sovereign_dedupe.py audit --preview 10` — clean output, no Python errors
3. Verify `--apply` is still gated behind `--confirm <PLAN_ID>` — never bypass this

**Never:**
- Add external dependencies — stdlib-only is an invariant
- Remove or weaken the `--apply + --confirm <PLAN_ID>` gate — this prevents accidental destruction
- Modify plan_sha256 tamper detection — it's the integrity guarantee
- Allow staging directory inside a target root — the guard exists for a reason
- Skip SHA-256 verification during execute

**Always:**
- Create rollback tag before any changes: `git tag pre-codex-$(date +%s)`
- Run `selftest` after every modification before shipping
- Keep the script single-file — no splitting into modules
- Preserve Ctrl+C kill-stop safety (writes durable manifest mid-run)
- DRY-RUN must remain the default — `--apply` opt-in only

---

## Commit Message Format

```
fix: <description>
feat: <description>
refactor: <description>
```

---

## Test Commands

```bash
python sovereign_dedupe.py selftest   # full test suite — must pass
python sovereign_dedupe.py audit      # smoke test — must produce output
```

---

## Coding Standards

- Python 3, stdlib-only — no imports outside the standard library
- Single-file architecture — `sovereign_dedupe.py` is the entire program
- All destructive operations behind explicit confirmation gates
- SHA-256 verification on every file operation
- Console output via the `Console` class — no raw `print()` for user-facing output
- Dataclasses for structured data (`@dataclass`)

---

## Protected Logic

- `--apply + --confirm <PLAN_ID>` gate — do not weaken or bypass
- `plan_sha256` tamper detection — do not remove
- Staging collision-safe filename logic — prevents overwrite
- Ctrl+C signal handler — must remain for kill-stop safety

---

*This file is read by Codex before every task. Keep it current.*
