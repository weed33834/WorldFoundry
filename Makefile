.PHONY: help install-core install-dev test-fast test-eval-core test-ux docs-check lint ruff-check format-check shell-check data-check compile-eval cli-check precommit precommit-install preflight

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
PRE_COMMIT ?= $(PYTHON) -m pre_commit
PYTHONPATH ?= .
WORLDFOUNDRY_EVAL ?= $(PYTHON) -m worldfoundry.cli
PREFLIGHT_PROFILE ?= all
PREFLIGHT_OUTPUT ?= tmp/preflight
CLI_CHECK_OUTPUT ?= tmp/ci-cli-check
RELEASE_HFD_ROOT ?= $(if $(WORLDFOUNDRY_HFD_ROOT),$(WORLDFOUNDRY_HFD_ROOT),$(HOME)/.cache/worldfoundry/checkpoints/hfd)
EVAL_CORE_CHECK_TESTS ?= \
	test/eval_core/test_api_contracts.py \
	test/eval_core/test_metric_registry.py \
	test/eval_core/test_task_yaml.py \
	test/eval_core/test_run_manifest.py \
	test/eval_core/test_public_namespace.py \
	test/eval_core/test_scorecard_snapshot.py \
	test/eval_core/test_contract_stability.py

help:
	@$(PYTHON) scripts/dev/check_dev_tools.py --help-targets

install-core:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e .
	$(PIP) install build pre-commit pytest PyYAML ruff

test-fast: test-eval-core test-ux docs-check

test-eval-core:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -m fast_eval_core $(EVAL_CORE_CHECK_TESTS)

test-ux:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) -q \
		test/eval_core/test_cli_ux.py \
		test/eval_core/test_catalog_discovery_output.py \
		test/eval_core/test_docs_quickstart.py

docs-check:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli --help >/dev/null
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli zoo benchmarks --json >/dev/null

lint:
	$(PYTHON) scripts/dev/check_dev_tools.py --lint

ruff-check:
	$(PYTHON) scripts/dev/check_dev_tools.py --ruff-check

format-check:
	$(PYTHON) scripts/dev/check_dev_tools.py --format-check

shell-check:
	$(PYTHON) scripts/dev/check_dev_tools.py --shell-check

data-check:
	$(PYTHON) scripts/dev/check_dev_tools.py --data-check

compile-eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m compileall -q worldfoundry/evaluation scripts

cli-check:
	rm -rf $(CLI_CHECK_OUTPUT)
	mkdir -p $(CLI_CHECK_OUTPUT)/input
	printf '%s\n' '{"sample_id":"ci-0001","status":"success","artifacts":{"video":{"uri":"$(CLI_CHECK_OUTPUT)/input/demo.mp4","kind":"video"}}}' > $(CLI_CHECK_OUTPUT)/input/results.jsonl
	: > $(CLI_CHECK_OUTPUT)/input/demo.mp4
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli evaluate \
		--mode existing-results \
		--results-path $(CLI_CHECK_OUTPUT)/input/results.jsonl \
		--output-dir $(CLI_CHECK_OUTPUT)/run \
		--benchmark-id ci-existing-results \
		--model-id ci-package-check \
		--metric artifact_count \
		--required-artifact video \
		--json

precommit:
	$(PRE_COMMIT) run -a

precommit-install:
	$(PRE_COMMIT) install

preflight:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m worldfoundry.cli preflight runtime \
		--profile $(PREFLIGHT_PROFILE) \
		--output-dir $(PREFLIGHT_OUTPUT) \
		--json
