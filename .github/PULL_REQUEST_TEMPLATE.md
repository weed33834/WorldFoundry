## Summary

<!-- Required: explain the change and why it belongs in WorldFoundry. -->

## User-Visible Behavior

<!-- Required: describe the public CLI, docs, catalog, scorecard, or runtime behavior users will see. Write "None" only for internal-only changes. -->

## Affected Pipeline / Benchmark

<!-- Required: name every touched pipeline, benchmark, runtime profile, or public entrypoint. Use N/A only for docs-only changes. -->

- Pipeline(s):
- Benchmark(s):
- Runtime profile(s):
- Public entrypoint(s):

## Change Type

- [ ] Bug fix
- [ ] Model integration
- [ ] Benchmark integration
- [ ] Pipeline/runtime change
- [ ] Documentation only
- [ ] Test/QA tooling

## Asset / API / GPU Requirements

<!-- Required: identify any large assets, API quota, secrets, official repos, simulators, or GPU requirements. Write "None" only if no external requirement exists. -->

- Downloads or large assets:
- API keys or quota:
- GPU / simulator requirements:
- Official repository or checkpoint assumptions:

## Commands Run

<!-- Required: paste the exact commands you ran and whether they passed, failed, or were skipped. -->

| Command | Result | Notes |
| --- | --- | --- |
| `conda run -n worldfoundry …` |  |  |

## Checkpoint / API Key Needs

<!-- Required: list any checkpoint paths, model weights, local caches, API keys, or environment variables needed to run this PR. Write "None" only if no external dependency is needed. Do not paste secrets. -->

- Checkpoints / weights:
- Required environment variables:
- API providers:
- Local cache assumptions:

## Sample Artifact Evidence

<!-- Required: attach or link representative outputs for visual generation, evaluation reports, logs, or screenshots. For docs-only or test-only changes, provide the relevant rendered page, command output, or explain why no artifact applies. -->

- Artifact path or link:
- What it demonstrates:
- Known limitations:

## Validation Matrix Status

<!-- Required: mark the smallest matrix that covers the change. Use "N/A" only when the row cannot apply. -->

| Area | Status | Evidence / Command |
| --- | --- | --- |
| Unit or focused regression | Not run / Pass / Fail / N/A |  |
| Real inference validation | Not run / Pass / Fail / N/A |  |
| Streaming or multi-turn path | Not run / Pass / Fail / N/A |  |
| Benchmark runner / metric path | Not run / Pass / Fail / N/A |  |
| Docs build or link check | Not run / Pass / Fail / N/A |  |
| GPU/API-key dependent path | Not run / Pass / Fail / N/A |  |

## Leaderboard Validity Impact

<!-- Required: explain whether this PR changes leaderboard eligibility, scorecard fields, readiness status, or validation evidence. -->

- Readiness status changed:
- `leaderboard_valid` impact:
- Scorecard/preflight evidence:
- Demo or contract-only limitations:

## Compatibility And Risk

- Backward compatibility impact:
- Expected resource cost:
- Failure modes or rollout concerns:

## Checklist

- [ ] I kept the change narrowly scoped.
- [ ] I did not commit secrets, API keys, large checkpoints, or generated cache files.
- [ ] I reused fixtures from `data/test_cases/` or assets from `data/benchmarks/` where practical.
- [ ] I updated docs when public behavior, install steps, or entry commands changed.
- [ ] I documented API/GPU/checkpoint/official-repo requirements or confirmed none are needed.
- [ ] I did not promote demo, contract-only, normalizer-only, API-blocked, or partial evidence to leaderboard validity.
