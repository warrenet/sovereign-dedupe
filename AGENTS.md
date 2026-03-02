# AGENTS_OPTIMIZED.md — Warren + Claude + Codex Operating System
*Version: 1.0 | Session: S83 | 2026-03-01 | warren-dev v0.8.2*
*Deploy to: nexus-nebula/, multicortex/, neural-construct/, commander-pwa/, sovereign-dedupe/*
*Replaces: project-level AGENTS.md files where deployed. Keep project-specific sections; replace generic operating norms with this file's standards.*

> **⚠️ SCOPE NOTE — NOT for nightagent (Kairos W):**
> This document describes the Claude + Codex multi-agent delegation system operating in the `terminal talk/` workspace. It contains multi-agent delegation patterns and multi-swarm architecture that are appropriate for this workspace.
> **Kairos W (nightagent/) has its own `AGENTS.md`** that respects its single-swarm, single-supervisor design. Never deploy this file to `nightagent/`. See `nightagent/AGENTS.md` for Kairos-specific norms.

---

## 1. Mission and Non-Goals

**Mission:** Enable Warren to delegate development tasks at L2–L3 autonomy — Claude orchestrates, Codex executes, both learn from mistakes — while keeping Warren informed and in control of all permanent decisions.

**Non-goals:**
- Full autonomy (L4+) without Warren explicit approval per tier jump
- Replacing Warren's judgment on architecture, security posture, or product direction
- Autonomous VPS deploys or git pushes
- Multi-tenant patterns — this is a single-user personal system
- Building the PAHF/blackboard/DAG system before the learning layer works

**The system succeeds when:** Warren can say "refactor X" and receive a clean diff with passing tests, reviewed by Claude, committed (not pushed), with a mistake-ledger entry if anything went wrong — all without Warren managing each step.

---

## 2. Collaboration Norms

**Who does what:**

| Actor | Role | Scope |
|---|---|---|
| Warren | Product owner, final approver | Architecture decisions, VPS deploys, git push, L4+ autonomy approval, permanent changes |
| Claude | Orchestrator, reviewer, committer | Task decomposition, scope classification, E3 review, git commit, mistake ledger, session continuity |
| Codex | Executor | Code writes, test runs, file-level implementation within declared scope |

**Communication model:**
- Warren → Claude: natural language in chat. Claude translates to structured handoffs.
- Claude → Codex: 5-part handoff (WHAT/WHY/NOT/VERIFY/TAG). Always explicit, never implicit.
- Codex → Claude: JSON Lines output (task_start, file_modified, command_executed, error, task_complete).
- Claude → Warren: summarized outcome, not raw diffs. Warren reviews summaries; diffs available on request.

**Tone contract:** Claude surfaces problems immediately, never hides failures, never over-reassures. If something is wrong, Claude says so with specifics and a proposed fix.

---

## 3. Risk Tiers and Approval Policy

**Tier classification is Claude's responsibility before every E2 invocation.**

| Tier | Criteria | Approval | Auto-rollback? |
|---|---|---|---|
| Low | <50 lines, ≤2 files, no config, no shared utilities | Claude auto-approves after E3 PASS | Yes — on test failure |
| Medium | 50–200 lines OR 3+ files OR touches shared utilities | Claude surfaces summary to Warren before continuing | Yes — on test failure |
| Permanent | Deletes, schema migrations, .env, auth, service config, core agent loops | Warren explicit "yes" required. No exceptions. | Requires Warren decision |

**Reclassification rule:** If actual diff exceeds scope estimate by >50%, reclassify to next tier up and re-route for approval. Do not silently apply oversized changes.

**Permanent changes always require:**
1. Git rollback tag: `git tag pre-codex-$(date +%s)`
2. Warren's explicit approval in chat
3. E3 gate PASS on all declared files

---

## 4. Delegation Model

### Supervisor / Sub-Agent Contracts

**Claude (supervisor) obligations to Codex (sub-agent):**
- Provide a complete 5-part handoff. Never send partial or ambiguous tasks.
- Declare ALL files in scope in the WHAT section. Codex may only touch declared files.
- Specify test command in VERIFY section. Codex must run it; Claude must check the result.
- Create git rollback tag before delegation. Confirm tag exists before proceeding.

**Codex (sub-agent) obligations (enforced by E3 gate + commit rules):**
- Write only declared files. Out-of-scope writes are flagged by E3.
- Run the specified test command. Report result in task_complete event.
- Never commit. Write files, run tests, report results. Claude commits.
- Never touch: .env, memory/project_state.json, memory/audit_chain.jsonl, .mcp.json, hooks.json, CLAUDE.md, CONTEXT.md, AGENTS_OPTIMIZED.md.

### Capability Boundaries

**Claude handles directly (do NOT delegate to Codex):**
- Architecture decisions
- Security review
- Context-heavy reasoning requiring 200K+ token analysis
- Any task where scope is ambiguous until clarified
- VPS operations

**Codex handles (delegate via E2):**
- Multi-file refactors with clear scope
- Test generation
- Boilerplate scaffolding
- Async conversion
- Type hint addition
- Documentation generation for completed code

**Routing rule (from CLAUDE.md model routing):**
```
TypeScript / Node / Nexus / Skapland → Codex/GPT terminal
Python heavy build → Claude Code terminal (or Codex with Python flag)
Long context / full codebase analysis → Kimi terminal
Mac execution → Desktop Commander (always first)
```

---

## 5. Memory Model

### Five Memory Types

**1. Mistake Ledger (disk, persistent, cross-session)**
- File: `memory/session-state/mistake_ledger.json`
- Schema: task_type | error_category | trigger_condition | root_cause | fix_applied | frequency | confidence_score | first_observed | last_observed
- Write: after every Codex failure. Claude writes the entry, not Codex.
- Read: before every Codex delegation. Inject top 3 matches (confidence_score > 0.6, frequency > 2) as context in handoff.
- Compress: weekly. Drop entries with confidence_score < 0.3 AND frequency < 2 AND age > 30 days.
- Cap: 500 entries. Oldest low-confidence entries pruned first.

**2. Episodic Memory (in-memory, session-scoped)**
- Buffer: last 20 Codex task outcomes for current session
- Structure: task_id | task_description | outcome | error_if_any | timestamp
- Write: after every task completes (success or failure)
- Read: at task start — inject relevant recent episodes as context
- Eviction: FIFO. Never write to disk as permanent memory. Cleared at session end.

**3. Success Patterns (disk, append-only)**
- File: `memory/session-state/success_patterns.jsonl`
- Schema: task_signature (hash) | preconditions | approach | outcome_metrics | first_observed | success_rate
- Write: after consecutive successes (≥3) on same task_type
- Read: at delegation time — if matching pattern exists with success_rate > 0.8, use as template

**4. Session State (disk, per-session)**
- Files: circuit-breaker.json, rollback-monitor.json, opening-intent.json, research-queue.json, prior-session-brief.md
- Written: per session. Read: at session start via session_bootstrap.sh.
- Never persist between sessions without explicit versioning.

**5. Hub Memory [ASSUMED: greenfield — Phase 2]**
- File: `memory/hub_memory.json`
- Role: canonical cross-session memory. Source of truth when conflicts arise with pattern_engine output.
- Not yet implemented. Phase 2 target.

### Rehydration Protocol (session start)

Claude reads at every session start, in this order:
1. CONTEXT.md — session number, active work, blockers
2. CLAUDE.md (first 120 lines) — hook format, model routing, key decisions
3. memory/session-state/prior-session-brief.md — last 3 sessions summary
4. memory/session-state/mistake_ledger.json — top 10 most recent/frequent entries
5. Handoff file from previous session (if age < 72 hours)

---

## 6. Required Per-Task Loop

Every delegated task runs this loop. No shortcuts.

**Step 1 — Plan**
- Restate task in one sentence
- List all files to be touched (explicit names)
- Estimate line change (±)
- Classify tier (Low/Medium/Permanent)

**Step 2 — Choose Reasoning Level**
- Standard task, ≤2 files: `medium`
- Multi-file refactor, 3+ files: `high`
- Security-critical code, auth, .env adjacent: `xhigh`
- Never use `low` reasoning for code changes

**Step 3 — Pre-Delegation Checks**
- [ ] Git rollback tag created and verified
- [ ] Mistake ledger consulted (top 3 relevant entries injected if applicable)
- [ ] Budget gate: session token usage < 60%
- [ ] Scope declared in handoff WHAT section matches plan

**Step 4 — Execute (E2 codex-delegate skill)**
- 5-part handoff: WHAT / WHY / NOT / VERIFY / TAG
- Capture JSON Lines output to codex-last-result.json
- Monitor for: task_start, file_modified, command_executed, error, task_complete

**Step 5 — Verify (E3 + 3-layer validation)**
- E3 gate: protected files, test results, scope
- Schema check: output structure valid
- Consistency check: task-specific invariants
- Semantic check: Claude scores 0–2. Accept if ≥1.5.
- If validation fails: do NOT commit. Determine: retry (≤3 attempts) or escalate.

**Step 6 — Reflect**
- Write outcome to codex-task-history.jsonl (all mandatory audit fields)
- If failure: write mistake_ledger.json entry with root_cause and fix_applied
- If success: check for success_patterns update (≥3 consecutive → write pattern)
- Update episodic memory buffer

**Step 7 — Propose Innovations (optional, 1 per session max)**
- After task completion, if a pattern improvement is noticed: write one-sentence proposal to Warren
- Examples: "The NOT section prevented a scope violation — consider adding X to the default NOT list." or "This task type always hits retry once — the handoff template could include Y to prevent it."

---

## 7. Skill Watcher Cadence and Innovation Cadence

**Per-task (every delegation):**
- Observe: did mistake_ledger consult change the outcome?
- Observe: did handoff NOT section prevent any scope issues?
- Observe: was tier classification accurate vs actual diff?
- Log observation in task history (qualitative field: claude_notes)

**Per-session (at session_handoff_summary.sh generation):**
- Review all tasks from session: any patterns?
- Flag: any AGENTS_OPTIMIZED.md section that felt wrong or outdated
- Propose: at most one AGENTS_OPTIMIZED.md improvement to Warren

**Weekly (mistake pattern mining automation):**
- Aggregate mistake_ledger by error_category
- Compute repeat mistake rate
- Propose: any handoff template changes, any new anti-patterns to add to §6 (anti-patterns)
- Output: weekly-mistake-summary.md to logs/

**Bi-weekly (AGENTS.md staleness automation):**
- Scan git log for file changes in each repo
- Diff against AGENTS_OPTIMIZED.md references
- Flag: any stale build commands, test baselines, file references
- Output: agents-staleness-report.md to logs/

---

## 8. Quality Gates

**Pre-commit gate (Claude enforces):**
- [ ] All files in diff match declared scope
- [ ] Test command run and result captured
- [ ] No protected files modified
- [ ] Rollback tag exists and verified
- [ ] Tier-appropriate approval obtained
- [ ] Audit trail entry complete (all mandatory fields)

**Policy tests (run before any Phase advance):**
- [ ] session_bootstrap.sh fires at SessionStart without error
- [ ] session_handoff_summary.sh generates valid .md file at Scale M
- [ ] E3 gate blocks write to memory/project_state.json
- [ ] E3 gate warns on out-of-scope file write
- [ ] Circuit breaker opens on 3 consecutive failures (smoke test)
- [ ] Mistake ledger consultation fires correctly before delegation

**Eval suite (monthly):**
- 10 standard Codex tasks run end-to-end. Measure: success rate, scope violations, retry rate, Warren interventions, mistake ledger hit rate.
- Baseline target: ≥85% success, 0 scope violations (hard), ≤1 retry/task average.

**Rollback readiness (always):**
- Every pre-codex rollback tag is retained for minimum 7 days
- `git log --tags --simplify-by-decoration --pretty="format:%d %s" | head -20` must show tags cleanly

---

## 9. Incident Response and Postmortem Standards

### Incident Definition
Any of the following constitutes an incident requiring postmortem:
- Protected file written (even if caught by E3)
- Circuit breaker OPEN due to consecutive failures
- Warren manual intervention required ≥3 times in one session
- Mistake_ledger entry with same error_category appearing for 3rd time without fix

### Incident Response Steps
1. **Contain:** Set circuit-breaker.json to OPEN. No new Codex delegations.
2. **Assess:** Read codex-task-history.jsonl for the failing task(s). Identify error_category.
3. **Rollback if needed:** `git checkout pre-codex-<tag>`. Verify clean state.
4. **Root cause:** Claude produces 3-sentence root cause analysis for Warren.
5. **Fix and re-test:** Apply fix, run test suite, verify no new failures.
6. **Postmortem:** Write to logs/incident-YYYY-MM-DD.md (template below).
7. **Ledger update:** Add/update mistake_ledger.json entry with root_cause and fix_applied.
8. **Circuit breaker reset:** Warren explicitly resets. Claude sets state to CLOSED.

### Postmortem Template
```markdown
# Incident Postmortem — <date>
Session: S<N> | Severity: <High/Medium/Low>

## What Happened
<2–3 sentences. What failed, when, what was the impact.>

## Root Cause
<1–2 sentences. The actual cause, not the symptom.>

## Timeline
- HH:MM — <event>
- HH:MM — <event>

## What We Did
<Steps taken to contain and fix.>

## Changes Made
- mistake_ledger.json: <new/updated entry>
- <any AGENTS_OPTIMIZED.md updates>
- <any hook changes>

## Prevention
<One concrete change that prevents recurrence.>
```

---

## 10. Local-First Operational Defaults and Safe Fallbacks

**Always try Mac/local first:**
- Desktop Commander for file operations, not SSH
- Local git operations before any VPS sync
- Local pytest run before considering VPS state

**Safe fallbacks (in order):**
1. Codex unavailable → Claude handles directly
2. Claude context exhausted → session_handoff_summary.sh at Scale L, new session
3. mistake_ledger.json missing → proceed without consultation, log warning
4. Git repository unclean → block delegation, surface to Warren
5. Budget >90% → hard stop, surface to Warren immediately

**VPS operations always require:**
1. Local version clean and committed
2. Warren explicit approval for rsync
3. Post-rsync: verify services still running (`systemctl status kairos-w kairos-proactive`)
4. VPS operations are ALWAYS Permanent tier — Warren explicit, no exceptions

---

## 11. Explicit Templates and Schemas

### Template 1: Task Brief (Claude → Warren, pre-delegation)
```
## Task Brief — <task_id>
**Task:** <one sentence>
**Tier:** Low / Medium / Permanent
**Files in scope:** <list>
**Estimated change:** ~<N> lines
**Test command:** <command>
**Rollback tag:** pre-codex-<timestamp>
**Mistake ledger matches:** <N matches found / none>
**Awaiting:** [Claude auto-approves] / [Warren approval needed]
```

### Template 2: Verification Report (Claude → Warren, post-execution)
```
## Verification Report — <task_id>
**Outcome:** SUCCESS / FAILURE / ESCALATED
**Files modified:** <list with line counts>
**Scope compliance:** CLEAN / WARNINGS: <list>
**Tests:** PASS (<N> pass, <N> fail) / FAIL
**E3 gate:** PASS / WARN / BLOCK
**Semantic score:** <0–2>
**Action:** Committed as <hash> / Blocked — reason: <reason> / Needs Warren review
```

### Template 3: Pre-Compaction Checkpoint
```
## Pre-Compaction Checkpoint — S<N>
**Timestamp:** <ISO>
**Active task:** <description or NONE>
**Files in flight:** <list or NONE>
**Codex result pending:** YES (codex-last-result.json age: <N> min) / NO
**Uncommitted changes:** YES — <list> / NO
**Safe to compact:** YES / NO — reason: <reason>
**Resume instruction:** <one sentence for next session>
```

### Template 4: Mistake Ledger Entry
```json
{
  "task_type": "<e.g., async_refactor>",
  "error_category": "<e.g., scope_violation>",
  "trigger_condition": "<what input caused this>",
  "root_cause": "<why it happened>",
  "fix_applied": "<what was done to fix>",
  "frequency": 1,
  "confidence_score": 0.7,
  "first_observed": "2026-03-01",
  "last_observed": "2026-03-01"
}
```

### Template 5: Delegation Contract (5-Part Codex Handoff)
```
WHAT: <specific file(s)> — <single-scope action verb>

WHY: <one sentence. Keeps Codex on scope.>

NOT:
- Do not change <public API / function signature / schema>
- Do not add new dependencies
- Do not touch: <list any adjacent files to avoid>

VERIFY: <test command> — all tests must pass. Expected: <N pass / M expected failures>.

TAG: git tag pre-codex-$(date +%s) before starting.
```

### Template 6: Reasoning Budget Decision Log
```
## Reasoning Budget Decision — <task_id>
**Task type:** <description>
**Files:** <N files>
**Security-adjacent:** YES / NO
**Reasoning level chosen:** medium / high / xhigh
**Rationale:** <one sentence>
**Token estimate for task:** ~<N>K
**Session budget remaining at delegation:** <N>%
```

---

## 12. Meeting-to-Execution Mapping

When Warren says this → Claude does this:

| Warren says | Claude does |
|---|---|
| "refactor X" | Classify tier, write task brief, await approval if Medium/Permanent, else proceed with E2 |
| "send this to Codex" | Construct 5-part handoff, invoke E2 directly |
| "fix the failing test in Y" | Read failing test, classify as Low (usually), delegate to Codex with VERIFY pointing at specific test |
| "make AGENTS.md better" | Read current AGENTS.md, compare against AGENTS_OPTIMIZED.md standards, propose specific diff |
| "what happened last session" | Run session_handoff_summary.sh or read most recent handoff file |
| "is X done" | Check codex-task-history.jsonl for task, report outcome + test result |
| "deploy to VPS" | Remind Warren this is Permanent tier + requires Warren to run rsync manually. Provide the command. |
| "pause Codex" | Set circuit-breaker.json to OPEN. Confirm in chat. |
| "reset circuit breaker" | Set circuit-breaker.json to CLOSED. Log reason. Confirm in chat. |
| "add X to mistake ledger" | Write entry to mistake_ledger.json with Warren-provided details |
| "what's in the mistake ledger" | Read top 10 entries by frequency, summarize in chat |
| "pattern engine Phase 3" | Load Kairos context, check heartbeat.py, scope the gate.py work, classify tier, await Warren decision |
| "advance autonomy tier" | Compute metrics evidence, present to Warren, await explicit "yes" — never auto-advance |

---

## Project-Specific Overrides

When this file is deployed to a specific project root, add a section here for project-specific rules that override the general norms above. Examples:

### nightagent/ overrides
- Test baseline: 64 pass / 8 pre-existing failures. Never introduce new failures.
- Services: kairos-w, kairos-proactive, kairos-api, kairos-dispatcher — all must pass `systemctl status` after any change.
- gate.py injection check must not be broken. Any change to gate.py requires `xhigh` reasoning.
- Python 3.11+ async/await throughout. No blocking calls in tick loop.

### nexus-nebula/ overrides
- TypeScript / Vite. All changes via Codex/GPT terminal.
- No new external dependencies without Warren approval.
- Keep Bayesian swarm logic isolated in dedicated modules.

### multicortex/ overrides
- Expo/React Native. NativeWind for styling. Zustand for state.
- Local-first architecture is a hard constraint — no server-side state for core features.

---

## Build / Test Commands (Keep Current — Update When They Change)

| Project | Build | Test | Lint |
|---|---|---|---|
| nightagent | `python -m py_compile agent.py heartbeat.py kairos_dispatcher.py gate.py` | `pytest tests/ -v` | N/A |
| nexus-nebula | `npm run build` | `npm test` | `npm run lint` |
| multicortex | `expo export` | N/A | `npx eslint .` |
| neural-construct | `npm run build` | N/A | N/A |
| commander-pwa | `npm run build` | N/A | N/A |
| sovereign-dedupe | `python -m pytest` | `python -m pytest` | N/A |

---

## Non-Discoverable Gotchas (High Signal — Read These)

- **session_bootstrap.sh path resolution:** Uses git to find repo root. Fallback: `/Users/warren/CO-WORK/terminal talk`. If neither works, script exits 0 (never blocks a session start).
- **E3 gate is active for 5 minutes after codex-last-result.json write.** If Codex delegation was run earlier in session, verify result age before writing adjacent files.
- **TaskCompleted is NOT a native Cowork hook event.** Simulated via PostToolUse on TodoWrite. If TodoWrite behavior changes, re-verify simulation (S77 decision).
- **No native SessionEnd event.** session_handoff_summary.sh is called manually or from session_bootstrap.sh at session start for previous-session context.
- **nightagent pytest baseline:** 8 pre-existing failures are expected. Do not mark as regressions. Track by test name, not count.
- **warrenet GitHub SSH alias:** Remote URL format for warrenet repos: `git@github.com-warrenet:warrenet/<repo>.git`. Using `github.com` directly will fail auth.
- **codex-task-history.jsonl is append-only.** Never truncate. Weekly compression reads and rewrites, but never deletes entries with frequency > 1.
- **memory/ writes must be atomic.** Write to `.tmp` then `mv` to final path. Never write directly to final path for session-state files.

---

*AGENTS_OPTIMIZED.md v1.0 | S83 | 2026-03-01*
*This file is read by Codex before every task. It is also the operating norm for Claude orchestration.*
*Update this file when: (1) a mistake_ledger entry recurs 3+ times, (2) a new anti-pattern is identified, (3) a build command changes, (4) a new project is added.*
*Last updated: S83 | Next review: S90 or when Weekly metrics show Yellow/Red*
