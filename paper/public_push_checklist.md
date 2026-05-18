# Public Push Checklist (minimal user-action version)

The README + paper/paper_draft.md have been pre-staged with
`<TODO_MAIN_URL>` / `<TODO_EXO_URL>` / `<TODO_GEM5_URL>` placeholders.
After pushing the 3 repos, run
`paper/scripts/finalize_public_push.sh <main> <exo> <gem5>` to
substitute the real URLs in one shot.

## Truly minimal user-action sequence

```bash
# 1. Auth once
gh auth login

# 2. Create + push each repo (one command each, gh handles the push)
cd /home/noah/project/riscv
gh repo create precision-routing-rvv-fa --public --source=. --remote=origin --push

cd exo
gh repo create exo-saturn-rvv --public --source=. --remote=origin --push

cd ../gem5
gh repo create gem5-saturn-fu --public --source=. --remote=origin --push

# 3. Back to main repo: substitute URLs + commit + push
cd /home/noah/project/riscv
./paper/scripts/finalize_public_push.sh \
  https://github.com/<YOUR_USERNAME>/precision-routing-rvv-fa \
  https://github.com/<YOUR_USERNAME>/exo-saturn-rvv \
  https://github.com/<YOUR_USERNAME>/gem5-saturn-fu
git add README.md paper/paper_draft.md
git commit -m "Public push: hosted-repo URLs in README + paper §9"
git push
```

That's it. ~5 commands, ~10 minutes hands-on.

## Pre-push state verified 2026-05-18. Three repos to push:

| Repo | Local path | Branch | Commits to push | Notes |
|---|---|---|---|---|
| Main project | `./` (a.k.a. `/home/noah/project/riscv/`) | `main` | 17 (`f99fb3d` → `4bdc40d`) | Saturn-FU + paper + microbenches + Exo schedule pass |
| Exo fork | `./exo/` | `main` | 10 on top of upstream `2f5472d` (`15d61db` → `b50ec24`) | SaturnRVV platform additions |
| gem5 fork | `./gem5/` | `stable` | 1 on top of upstream `c8222cc` (`841d376`) | RISC-V decoder + FU-latency wiring for Saturn customs |

All three repos audited 2026-05-18:
- **Main repo**: 14 Mo 8 result docs sanitized (`/home/noah/project/riscv/` → `./`; `/tmp/bootlin-14/...` → `/path/to/bootlin-riscv64-gcc14`). Final `git ls-files | xargs grep -l "/home/noah"` returns clean (only `y1_handoff_kickoff.md` mentions the string in commit-message + verification-statement context, not as a leaked path).
- **Exo fork**: clean (no `/home/noah` paths in our 10 commits).
- **gem5 fork**: clean (no `/home/noah` paths in our 1 commit; gem5 LICENSE inherited from upstream BSD-3-Clause).

---

## Step 1 — LICENSE copyright placeholder (user action)

`./LICENSE` currently reads:

> Copyright (c) 2026, **The Precision-Routing RVV Flash Attention Project Authors**.

Edit to your preferred attribution before pushing. Options:

- Your real name: `Copyright (c) 2026, <Your Name>.`
- Anonymized for double-blind submission: leave as-is.
- Institution (only if institutional approval secured): `Copyright (c) 2026, <Your Name> and <Institution>.`

Commit the edit:

```bash
cd ./
$EDITOR LICENSE
git add LICENSE
git commit -m "License: set copyright holder"
```

Exo fork inherits Exo's upstream LICENSE (MIT/Apache — check
`exo/LICENSE` if present, otherwise see upstream `exo-lang/exo`).
gem5 fork inherits gem5's upstream BSD-3-Clause LICENSE
(`gem5/LICENSE.txt`). Neither needs a new LICENSE file for our
patch contributions, but Saturn-attributable code in either fork
inherits the upstream license terms.

## Step 2 — Confirm git author identity (user action)

Current author config (from gem5 commit `841d376`):

- Name: `noyaboy`
- Email: `science103555@gmail.com`

Confirm this matches your preferred public identity. If not, set
per-repo overrides BEFORE pushing:

```bash
git config user.name  "<Your Name>"
git config user.email "<your-public-email>"
```

(Existing commits keep their original author — this only affects
new commits. If you want to retroactively rewrite the historical
author across the 17-commit main branch, use `git filter-branch`
or `git filter-repo`; this rewrites SHA hashes, so only do it
BEFORE the first push to a public host.)

## Step 3 — Pick a hosting choice (user action)

- **GitHub** (most common). `gh repo create --public` from each repo
  directory. Suggested names: `precision-routing-rvv-fa`,
  `exo-saturn-rvv`, `gem5-saturn-fu`.
- **Codeberg** (free, EU-hosted, Gitea backend). Manual repo create
  + `git remote add origin git@codeberg.org:...`.
- **SourceHut** (paid, mailing-list-flavored). Manual.
- **Your university's GitLab / institutional host**.

The paper's "open-source under permissive license" claim doesn't
require any specific host; pick whatever fits your double-blind /
attribution constraints.

## Step 4 — Push commands

After setting LICENSE + identity + remote, for each repo:

```bash
# Main project
cd ./
git remote add origin <main-repo-url>
git push -u origin main

# Exo fork (push our branch on top of upstream's main)
cd exo
git remote add origin <exo-fork-url>
git push -u origin main

# gem5 fork (we're on `stable`)
cd ../gem5
git remote add origin <gem5-fork-url>
git push -u origin stable
```

## Step 5 — Post-push verification

1. Each repo's `README.md` (main repo) / `README` (Exo upstream) /
   `README.md` (gem5 upstream) renders correctly on the host.
2. The 17-commit history on the main repo's `main` branch shows
   the Y1 arc (`f99fb3d` Initial → `4bdc40d` Paper figures).
3. The Mo 8 result docs (`paper/mo8_step*_results.md`) render and
   contain no `/home/noah` paths.
4. The LICENSE shows your chosen copyright line, not the placeholder.
5. From a fresh clone, the reproduction recipe in `README.md`
   works (at minimum: `sbt test` on `saturn-fu/`, the gem5 + bench
   build pattern).
6. Cross-reference URLs: edit `paper/paper_draft.md` to replace
   `./` references with the hosted-repo URLs if needed for the
   submission version.

## Step 6 — Update paper Abstract + README with hosted URLs

After pushing, update:

- `./README.md`: add the hosted-repo URL list (main + exo + gem5)
  in a "Repositories" section.
- `./paper/paper_draft.md` Abstract: the current "All RTL, gem5
  patches, kernels, and Exo declarations + the scheduling pass
  open-source" claim is fulfilled by the push; consider adding a
  short URL footnote at the abstract end (or in §9) listing the
  three repo URLs.

Suggested commit message:

```
Public push: hosted-repo URLs in README + paper §9
```

---

## What's intentionally NOT in this checklist

- `gh pr create` workflow: this isn't a PR; it's a first push of
  three independent repos.
- Force-pushing or branch-renaming history: only safe BEFORE
  first push. If you want historical author rewriting (Step 2),
  do it now, not after.
- Tagging releases: defer until paper acceptance / camera-ready.
