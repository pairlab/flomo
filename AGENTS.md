## Coding Principles

🚨 **FAIL HARD**: Do not gracefully handle failures or add fallbacks that hide bugs. When something doesn't work, let it fail immediately and visibly. No `config.get()` with hardcoded defaults outside of config. No `.get()` with defaults, `try/except` that swallows errors, or graceful degradation paths. Avoid default arguments for functions.

🚨 **KEEP IT SIMPLE**: Keep code as simple as possible while retaining functionality. Clean abstractions that are legible and easy to debug. Constantly assess whether you are overcomplicating things.

🚨 **TENSOR MANIPULATION**: Use einops (`repeat`, `rearrange`, etc.) to manipulate tensors. Annotate tensor shapes with variable names inline. Avoid `-1` indexing (unintended behavior). Avoid `squeeze()` (inconsistent behavior). When explaining tensor operations, show the actual shapes (with variable names and numbers) in einops rearrange format — not prose.

🚨 **AVOID LARGE FUNCTIONS**: If a function exceeds ~200 lines, simplify: delete superfluous logic or split into smaller functions.

🚨 **SHORT VARIABLE NAMES**: New variable names should be as short as possible while remaining legible. No silly abbreviations. Good: `det_ce` instead of `deterministic_cross_entropy_loss`.

🚨 **BE SKEPTICAL**: Do not declare victory after a fix. 90% of the time the fix doesn't exactly work. After a fix, verify it actually works. Read the diff, check edge cases, hunt for inconsistencies and logic mistakes. Be decisive about approach, but open to simpler solutions.

🚨 **NO IRREVERSIBLE CHANGES**: Do not delete files, make git commits, or make irreversible changes without explicit confirmation.

🚨 **REDUCE THE DIFF**: After a fix works, actively reduce the diff. Remove sanity checks, extra logging, and defensive code that was added during debugging.

🚨 **SINGLE SOURCE OF TRUTH FOR CONFIG**: Do not duplicate config values across files. If a value is needed in multiple places, reference it from one canonical location. Do not create separate parameters that should share a single value (e.g. action_dim appearing in two config sections).

🚨 **SHARED MODULES OVER SEPARATE ONES**: When two components need similar computation, prefer one module with branching heads over two separate modules. Do not create separate networks when weight-sharing is the intent.

🚨 **SURGICAL EDITS, NOT WHOLESALE REPLACEMENT**: When asked to fix or remove something, change only the specific part — do not truncate or rewrite surrounding content.

🚨 **FIX ROOT CAUSE, NOT SYMPTOMS**: Do not propose workarounds when the underlying issue is fixable. Fix the environment, not the caller.

🚨 **CORE CODE NEEDS EXTRA SCRUTINY**: When modifying shared/core code paths (train.py, model forward passes, etc), explicitly verify no breaking changes before declaring done.

🚨 **PROVE IT WORKS**: When setting up infrastructure, actively test and show proof. "Looks like it should work" is not acceptable.

## Project

- `README.md` describes the motion-token pipeline, configs, entry points, and layout.
- Requires a CUDA GPU. `Wan2.2/` and `dreamzero/` (plus `SpaTrackerV2/` and `vggt/` for track preprocessing) are repo-root clones, not pip packages. Run commands from the repo root.
