#!/usr/bin/env python3
"""Generate Fumadocs benchmark subpages from catalog YAML manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_ROOT = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "catalog"
TASK_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "tasks" / "external"
OUT_DIR = REPO_ROOT / "docs" / "fumadocs" / "content" / "docs" / "evaluation" / "benchmark-hub"
CATALOG_STATUS_OUT = REPO_ROOT / "docs" / "fumadocs" / "lib" / "benchmark-catalog-status.json"
INTROS_PATH = Path(__file__).resolve().parent / "benchmark_readme_intros.json"
METRIC_FOCUS_PATH = Path(__file__).resolve().parent / "benchmark_metric_focus.json"
ENV_SETUP_PATH = Path(__file__).resolve().parent / "benchmark_environment_setup.json"
RUNTIME_PROFILE_DIR = REPO_ROOT / "worldfoundry" / "data" / "benchmarks" / "runtime_profiles" / "official"

CATEGORY_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "Embodied AI",
        "Embodied AI",
        (
            "ai2thor",
            "behavior1k",
            "bridgedata-v2",
            "calvin",
            "kinetix",
            "libero",
            "libero-mem",
            "libero-para",
            "libero-plus",
            "libero-pro",
            "maniskill",
            "maniskill2",
            "metaworld",
            "mikasa",
            "molmospaces",
            "rlbench",
            "robocasa",
            "robocerebra",
            "robomme",
            "robotwin",
            "simpler-env",
            "vlabench",
        ),
    ),
    (
        "Video Generation",
        "Video Generation",
        (
            "vbench",
            "vbench-plus-plus",
            "video-bench",
            "vmbench",
            "t2v-compbench",
            "videoscore",
            "chronomagic-bench",
            "evalcrafter",
            "fetv",
            "aigcbench",
            "mirabench",
            "devil-dynamics",
            "genai-bench",
            "t2v-safety-bench",
        ),
    ),
    (
        "World Models",
        "World Models",
        (
            "vbench-2.0",
            "worldmodelbench",
            "worldscore",
            "worldbench",
            "wbench",
            "iworld-bench",
            "camerabench",
            "t2vworldbench",
            "videoverse",
            "physvidbench",
            "phygenbench",
            "videophy",
            "videophy2",
            "physics-iq",
            "ipv-bench",
            "videoscience-bench",
            "phyeduvideo",
            "worldarena",
            "world-in-world",
            "phyground",
            "phyfps-bench-gen",
            "ewmbench",
        ),
    ),
)

CATEGORY_BY_ID = {
    benchmark_id: (label_en, label_zh)
    for label_en, label_zh, ids in CATEGORY_GROUPS
    for benchmark_id in ids
}

def _load_readme_intros() -> dict[str, dict[str, str]]:
    if not INTROS_PATH.is_file():
        return {}
    payload = json.loads(INTROS_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


README_INTROS = _load_readme_intros()


def _load_metric_focus() -> dict[str, dict[str, dict[str, str]]]:
    if not METRIC_FOCUS_PATH.is_file():
        return {}
    payload = json.loads(METRIC_FOCUS_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


METRIC_FOCUS = _load_metric_focus()

GENERIC_DESC_MARKERS = (
    "official vbench dimension score",
    "normalized benchmark score",
    "mean over available",
)


def _load_environment_setup() -> dict[str, dict[str, str]]:
    if not ENV_SETUP_PATH.is_file():
        return {}
    payload = json.loads(ENV_SETUP_PATH.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


ENVIRONMENT_SETUP = _load_environment_setup()

FORMULA_BINDINGS_PATH = (
    REPO_ROOT / "worldfoundry" / "evaluation" / "tasks" / "execution" / "runners" / "_benchmark_metrics" / "bindings.py"
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _catalog_path(benchmark_id: str) -> Path | None:
    for shard in ("embodied", "video"):
        candidate = CATALOG_ROOT / shard / f"{benchmark_id}.yaml"
        if candidate.is_file():
            return candidate
    return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if item]
    return []


def _metric_ids(catalog: Mapping[str, Any]) -> list[str]:
    metrics = catalog.get("metrics")
    if not isinstance(metrics, list):
        return []
    ids: list[str] = []
    for item in metrics:
        if isinstance(item, Mapping) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _primary_metrics(catalog: Mapping[str, Any]) -> list[str]:
    metrics = catalog.get("metrics")
    if not isinstance(metrics, list):
        return []
    primary = [str(item["id"]) for item in metrics if isinstance(item, Mapping) and item.get("primary")]
    return primary or _metric_ids(catalog)[:4]


def _formula_metrics(benchmark_id: str) -> list[str]:
    text = FORMULA_BINDINGS_PATH.read_text(encoding="utf-8")
    metrics: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(f'("{benchmark_id}",'):
            continue
        parts = stripped.split('"')
        if len(parts) >= 4:
            metrics.append(parts[3])
    return list(dict.fromkeys(metrics))


def _escape_mdx(value: str) -> str:
    value = value.replace("{", "\\{")
    return re.sub(r"(?<![`])<([^>`]+)>", r"`<\1>`", value)


def _runtime_kind(catalog: Mapping[str, Any]) -> str:
    runner = catalog.get("runner")
    if not isinstance(runner, Mapping):
        return ""
    runtime = runner.get("runtime")
    if not isinstance(runtime, Mapping):
        return ""
    return str(runtime.get("kind") or "")


def _uses_generated_artifacts(catalog: Mapping[str, Any]) -> bool:
    layout = catalog.get("artifact_layout")
    if isinstance(layout, Mapping) and layout.get("generated_artifact_dir_env"):
        return True
    runner = catalog.get("runner")
    if isinstance(runner, Mapping):
        runtime = runner.get("runtime")
        if isinstance(runtime, Mapping) and runtime.get("generated_artifact_dir_env"):
            return True
    return any("GENERATED_ARTIFACT" in item for item in _string_list(catalog.get("requires")))


def _is_embodied_benchmark(catalog: Mapping[str, Any]) -> bool:
    kinds = {item.lower() for item in _string_list(catalog.get("benchmark_kind"))}
    domains = {item.lower() for item in _string_list(catalog.get("domains"))}
    embodied_tokens = {
        "vla",
        "robot-manipulation",
        "embodied-evaluation",
        "embodied",
        "manipulation",
    }
    return bool(kinds & embodied_tokens or domains & embodied_tokens)


def _requires_hosted_api(catalog: Mapping[str, Any]) -> bool:
    surface = catalog.get("evaluation_surface")
    if not isinstance(surface, Mapping):
        return False
    requires_api = str(surface.get("requires_api") or "").strip().lower()
    if not requires_api or requires_api in {"false", "no", "none"}:
        return False
    return "no hosted" not in requires_api and "local" not in requires_api[:20]


def _metric_aggregation_phrase(item: Mapping[str, Any], *, locale: str) -> str:
    metric_id = str(item.get("id") or "")
    name = str(item.get("name") or metric_id)
    aggregator = str(item.get("aggregator") or "mean").replace("_", " ")
    is_aggregate = metric_id.endswith("_average") or "aggregate" in name.lower() or name.lower().startswith("overall")
    if not is_aggregate:
        return ""
    if locale == "zh":
        return f"；各样本分数按 {aggregator} 汇总"
    return f"; per-sample scores are aggregated with {aggregator}"


def _is_generic_description(description: str) -> bool:
    lowered = description.lower()
    return any(marker in lowered for marker in GENERIC_DESC_MARKERS)


def _metric_focus(item: Mapping[str, Any], *, benchmark_id: str, locale: str) -> str:
    metric_id = str(item.get("id") or "")
    name = str(item.get("name") or metric_id or "Metric")
    description = str(item.get("description") or "").strip()
    lowered_id = metric_id.lower()
    lowered_name = name.lower()

    override = METRIC_FOCUS.get(benchmark_id, {}).get(metric_id, {}).get(locale)
    if override:
        return override

    if description and not _is_generic_description(description):
        return description

    if lowered_id.endswith("_average") or lowered_name.startswith("overall") or "aggregate" in lowered_name:
        if locale == "zh":
            return f"汇总 {name} 相关子指标后的 benchmark 总分。"
        return f"Aggregate benchmark score summarizing the underlying {name} component metrics."

    if "success" in lowered_id or "success" in lowered_name:
        if locale == "zh":
            return "episode / task 是否成功完成的比例或成功率。"
        return "Share of episodes or tasks that complete successfully."

    if "accuracy" in lowered_id:
        if locale == "zh":
            return "预测动作、状态或结果相对 benchmark 成功标准的正确性。"
        return "Correctness of predicted actions, states, or outcomes against the benchmark success criterion."

    if "reward" in lowered_id:
        if locale == "zh":
            return f"{name}：仿真 rollout 中累计的任务 reward 信号。"
        return f"{name}: cumulative task reward signal from simulator rollouts."

    if locale == "zh":
        return f"官方 {name} 指标在 benchmark 协议中的计分结果。"
    return f"Official {name} score under the benchmark evaluation protocol."


def _benchmark_eval_method(catalog: Mapping[str, Any], *, locale: str) -> str:
    runtime_kind = _runtime_kind(catalog)
    embodied = _is_embodied_benchmark(catalog)
    generated = _uses_generated_artifacts(catalog)
    hosted_api = _requires_hosted_api(catalog)

    if locale == "zh":
        methods = {
            "native_closed_loop_simulator": "在 benchmark 仿真环境中对候选 policy 做闭环 rollout，并记录 episode 结果。",
            "external_official_results_runner": "读取官方或上游 rollout / eval 结果并归一化为 WorldFoundry scorecard；完整复现通常仍依赖外部仿真资产。",
            "in_tree_official_runtime": "使用仓内 official metric runtime，对 caller 提供的生成产物运行官方指标栈（本地 checkpoint / evaluator）。",
            "in_tree_official_source": "通过仓内 vendored official pipeline，对生成产物执行或归一化官方 benchmark 流程。",
            "in_tree_official_judge_runtime": "使用仓内 official judge / VLM runtime，按 prompt 对生成视频打分。",
            "in_tree_judge_runtime": "使用仓内 judge 模型，对生成视频回答 benchmark 特定问题并计分。",
            "in_tree_artifact_metric_importer": "导入预计算的 metric table 或 score artifact，并归一化为 benchmark scorecard。",
            "in_tree_artifact_evaluator": "使用仓内 evaluator，按 official task protocol 评估 caller 提供的 artifact bundle。",
            "in_tree_official_runner": "调用仓内 official runner，按 task manifest 对生成产物执行 benchmark。",
        }
        method = methods.get(
            runtime_kind,
            "按 catalog task manifest 执行 benchmark，并写出 normalized scorecard。",
        )
        if generated and not embodied:
            method += " 候选模型输出从 generated-artifact 目录读取。"
        elif embodied:
            method += " 指标来自仿真 rollout 或官方结果文件中的 episode 级记录。"
        if hosted_api:
            method += " 部分维度可能依赖 hosted judge / API。"
        return method

    methods = {
        "native_closed_loop_simulator": "Runs closed-loop simulator rollouts with the candidate policy and records episode-level outcomes.",
        "external_official_results_runner": "Normalizes official or upstream rollout/eval result files into WorldFoundry scorecards; full replay may still require external simulator assets.",
        "in_tree_official_runtime": "Runs the in-tree official metric runtime on caller-supplied generated artifacts using local checkpoints and evaluators.",
        "in_tree_official_source": "Executes or normalizes the official benchmark pipeline from in-tree vendored sources on generated outputs.",
        "in_tree_official_judge_runtime": "Uses the in-tree official judge/VLM runtime to score generated videos per prompt.",
        "in_tree_judge_runtime": "Uses in-tree judge models to answer benchmark-specific questions about generated videos.",
        "in_tree_artifact_metric_importer": "Imports precomputed metric tables or score artifacts and normalizes them into the benchmark scorecard.",
        "in_tree_artifact_evaluator": "Evaluates supplied artifact bundles with in-tree evaluators tied to the official task protocol.",
        "in_tree_official_runner": "Invokes the in-tree official runner on generated artifacts following the benchmark task manifest.",
    }
    method = methods.get(
        runtime_kind,
        "Executes the benchmark through the catalog task manifest, then writes a normalized scorecard.",
    )
    if generated and not embodied:
        method += " Candidate outputs are read from the generated-artifact directory."
    elif embodied:
        method += " Metrics come from simulator rollouts or episode records in official result files."
    if hosted_api:
        method += " Some dimensions may rely on a hosted judge/API."
    return method


def _benchmark_eval_method_short(catalog: Mapping[str, Any], *, locale: str) -> str:
    runtime_kind = _runtime_kind(catalog)
    embodied = _is_embodied_benchmark(catalog)
    generated = _uses_generated_artifacts(catalog)
    hosted_api = _requires_hosted_api(catalog)

    if locale == "zh":
        if embodied:
            method = "闭环仿真 rollout，或归一化官方 rollout 结果"
        elif runtime_kind.endswith("_judge_runtime"):
            method = "仓内 judge / VLM 对生成视频逐 prompt 评分"
        elif "importer" in runtime_kind:
            method = "导入预计算 metric 并归一化为 scorecard"
        elif "evaluator" in runtime_kind:
            method = "仓内 evaluator 按 task protocol 评估 artifact bundle"
        elif generated:
            method = "仓内 official runtime 对生成视频逐 prompt 评分"
        else:
            method = "按 task manifest 执行 benchmark 并归一化 scorecard"
        if hosted_api:
            method += "；部分维度可能调用 hosted API"
        return method

    if embodied:
        method = "Closed-loop simulator rollouts, or normalization of official rollout results"
    elif runtime_kind.endswith("_judge_runtime"):
        method = "In-tree judge/VLM scores generated videos per prompt"
    elif "importer" in runtime_kind:
        method = "Imports precomputed metrics and normalizes them into a scorecard"
    elif "evaluator" in runtime_kind:
        method = "In-tree evaluator scores artifact bundles per the task protocol"
    elif generated:
        method = "In-tree official runtime scores generated videos per prompt"
    else:
        method = "Runs the task manifest benchmark flow and normalizes a scorecard"
    if hosted_api:
        method += "; some dimensions may call a hosted API"
    return method


def _metric_table_description(item: Mapping[str, Any], *, benchmark_id: str, locale: str) -> str:
    return _escape_mdx(_metric_focus(item, benchmark_id=benchmark_id, locale=locale))


def _format_command_block(command: Any) -> str:
    if isinstance(command, str):
        return command.strip()
    if isinstance(command, (list, tuple)):
        return " ".join(str(item) for item in command)
    return ""


def _build_run_section(catalog: Mapping[str, Any], benchmark_id: str, *, locale: str) -> str:
    runner = catalog.get("runner")
    runner_cmd = ""
    validation_cmd = _format_command_block(catalog.get("validation_command"))
    if isinstance(runner, Mapping):
        runner_cmd = _format_command_block(runner.get("run_command") or runner.get("validation_command"))

    if locale == "zh":
        intro = "准备好该 benchmark 页面列出的资产和候选产物后，使用下面的公开运行入口。"
        artifact_label = "设置候选产物目录"
        normalizer_label = "导入已有官方结果"
        official_run_label = "从生成产物重新打分"
        direct_label = "直接调用仓内 runner"
    else:
        intro = "After preparing the assets and candidate artifacts listed on this page, use the public run entry points below."
        artifact_label = "Set Candidate Artifacts"
        normalizer_label = "Import Official Results"
        official_run_label = "Score Generated Artifacts"
        direct_label = "Direct In-Tree Runner"

    artifact_cmd = "export WORLDFOUNDRY_GENERATED_ARTIFACT_DIR=/path/to/generated/artifacts"
    normalizer_cmd = "\n".join(
        [
            "worldfoundry-eval zoo benchmark-run \\",
            f"  --benchmark-id {benchmark_id} \\",
            "  --mode normalizer \\",
            "  --official-results-path /path/to/official/results-or-report \\",
            '  --generated-artifact-dir "${WORLDFOUNDRY_GENERATED_ARTIFACT_DIR}" \\',
            f"  --output-dir tmp/{benchmark_id}/normalizer \\",
            "  --json",
        ]
    )
    official_run_cmd = "\n".join(
        [
            "worldfoundry-eval zoo benchmark-run \\",
            f"  --benchmark-id {benchmark_id} \\",
            "  --mode official-run \\",
            '  --generated-artifact-dir "${WORLDFOUNDRY_GENERATED_ARTIFACT_DIR}" \\',
            f"  --output-dir tmp/{benchmark_id}/official-run \\",
            "  --json",
        ]
    )

    sections = [
        intro,
        "",
        f"### {artifact_label}",
        "",
        "```bash",
        artifact_cmd,
        "```",
        "",
        f"### {normalizer_label}",
        "",
        "```bash",
        normalizer_cmd,
        "```",
        "",
        f"### {official_run_label}",
        "",
        "```bash",
        official_run_cmd,
        "```",
    ]
    direct_cmd = runner_cmd or validation_cmd
    if direct_cmd:
        sections.extend(["", f"### {direct_label}", "", "```bash", direct_cmd, "```"])
    return "\n".join(sections)


def _build_artifacts_section(catalog: Mapping[str, Any], *, locale: str) -> str:
    requires = _string_list(catalog.get("requires"))
    runner = catalog.get("runner")
    expected: list[str] = []
    if isinstance(runner, Mapping):
        expected = _string_list(runner.get("expected_artifacts"))
    artifact_layout = catalog.get("artifact_layout")
    env_vars: list[str] = []
    if isinstance(artifact_layout, Mapping) and artifact_layout.get("generated_artifact_dir_env"):
        env_vars.append(str(artifact_layout["generated_artifact_dir_env"]))

    if locale == "zh":
        title_req = "Inputs"
        title_env = "Environment"
        title_out = "Outputs"
        empty = "当前 catalog 未声明额外条目。"
    else:
        title_req = "Inputs"
        title_env = "Environment"
        title_out = "Outputs"
        empty = "The current catalog does not declare extra items."

    lines: list[str] = []
    if requires:
        lines.extend([f"### {title_req}", ""])
        lines.extend(f"- {_escape_mdx(item)}" for item in requires)
        lines.append("")
    if env_vars:
        lines.extend([f"### {title_env}", ""])
        lines.extend(f"- `{item}`" for item in env_vars)
        lines.append("")
    if expected:
        lines.extend([f"### {title_out}", ""])
        lines.extend(f"- `{item}`" for item in expected)
        lines.append("")
    if not lines:
        return empty
    return "\n".join(lines).rstrip()


def _intro_paragraph(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text.split("\n\n", 1)[0].strip()


def _readme_blurb(benchmark_id: str, catalog: Mapping[str, Any], *, locale: str) -> str:
    custom = README_INTROS.get(benchmark_id, {}).get(locale)
    if custom:
        return _intro_paragraph(custom)
    task_path = TASK_DIR / f"{benchmark_id}.yaml"
    if task_path.is_file():
        task = _load_yaml(task_path)
        task_desc = str(task.get("description") or "").strip()
        if task_desc:
            return _escape_mdx(_intro_paragraph(task_desc))
    name = str(catalog.get("name") or benchmark_id)
    if locale == "zh":
        return f"{name} 的指标、依赖与运行方式见下方说明。"
    return f"See the sections below for {name} metrics, requirements, and run commands."


def _build_about_section(catalog: Mapping[str, Any], benchmark_id: str, *, locale: str) -> str:
    return _readme_blurb(benchmark_id, catalog, locale=locale)


def _build_evaluates_section(catalog: Mapping[str, Any], benchmark_id: str, *, locale: str) -> str:
    metrics = catalog.get("metrics")
    formula_metrics = _formula_metrics(benchmark_id)
    task_path = TASK_DIR / f"{benchmark_id}.yaml"
    task_desc = ""
    if task_path.is_file():
        task = _load_yaml(task_path)
        task_desc = str(task.get("description") or "")

    if locale == "zh":
        title = "指标"
        task_label = "Task YAML"
        primary_label = "Primary"
        formula_label = "Formula bindings"
        desc_header = "关注点"
    else:
        title = "Metrics"
        task_label = "Task YAML"
        primary_label = "Primary"
        formula_label = "Formula bindings"
        desc_header = "Focus"

    lines = [f"## {title}", ""]
    if task_desc:
        lines.extend([_escape_mdx(task_desc), ""])
    lines.extend([f"### {task_label}", "", f"`worldfoundry/data/benchmarks/tasks/external/{benchmark_id}.yaml`", ""])
    primary = _primary_metrics(catalog)
    if primary:
        lines.extend([f"### {primary_label}", ""])
        lines.extend(f"- `{metric}`" for metric in primary)
        lines.append("")
    if isinstance(metrics, list) and metrics:
        lines.extend([f"| Metric | {desc_header} |", "| --- | --- |"])
        for item in metrics:
            if not isinstance(item, Mapping):
                continue
            metric_id = str(item.get("id") or "")
            desc = _metric_table_description(item, benchmark_id=benchmark_id, locale=locale)
            lines.append(f"| `{metric_id}` | {desc} |")
        lines.append("")
    if formula_metrics:
        lines.extend([f"### {formula_label}", ""])
        lines.extend(
            f"- `worldfoundry/evaluation/tasks/execution/runners/_benchmark_metrics/` → `{metric}` (`FORMULA_EVALUATOR_BINDINGS`)"
            for metric in formula_metrics
        )
        lines.append("")
    return "\n".join(lines).rstrip()


def _runtime_profile(benchmark_id: str) -> dict[str, Any]:
    profile_path = RUNTIME_PROFILE_DIR / f"{benchmark_id}.yaml"
    if not profile_path.is_file():
        return {}
    return _load_yaml(profile_path)


def _environment_setup_body(catalog: Mapping[str, Any], benchmark_id: str, *, locale: str) -> str:
    override = ENVIRONMENT_SETUP.get(benchmark_id, {})
    if isinstance(override, Mapping):
        custom = override.get(locale)
        if isinstance(custom, str) and custom.strip():
            return custom.strip()

    profile = _runtime_profile(benchmark_id)
    if not profile:
        return ""

    env_id = str(profile.get("environment_id") or "worldfoundry-unified-cu128")
    needs_new_env = bool(profile.get("needs_new_env"))
    preflight = _format_command_block(profile.get("preflight_command"))
    base_deps = _string_list(catalog.get("base_model_dependencies") or profile.get("base_model_dependencies"))
    checkpoint_refs = _string_list(catalog.get("checkpoint_refs"))
    required_env = _string_list(profile.get("required_env") or catalog.get("requires"))
    profile_notes = _string_list(profile.get("notes"))
    preflight_notes: list[str] = []
    preflight_block = catalog.get("base_model_dependency_preflight")
    if isinstance(preflight_block, Mapping):
        preflight_notes = _string_list(preflight_block.get("notes"))

    dataset = catalog.get("dataset")
    dataset_reason = ""
    if isinstance(dataset, Mapping):
        if dataset.get("not_applicable"):
            dataset_reason = str(dataset.get("reason") or "").strip()
        elif dataset.get("reason"):
            dataset_reason = str(dataset.get("reason") or "").strip()

    assets = catalog.get("benchmark_assets")
    prompt_ref = ""
    if isinstance(assets, Mapping):
        prompt_suite = assets.get("prompt_suite")
        if isinstance(prompt_suite, Mapping) and prompt_suite.get("path_or_ref"):
            prompt_ref = str(prompt_suite["path_or_ref"])

    artifact_env = ""
    layout = catalog.get("artifact_layout")
    if isinstance(layout, Mapping) and layout.get("generated_artifact_dir_env"):
        artifact_env = str(layout["generated_artifact_dir_env"])

    required_assets = profile.get("required_assets")
    asset_paths: list[str] = []
    if isinstance(required_assets, Mapping):
        for key in ("local_dataset_paths", "local_repo_paths", "in_tree_source_paths"):
            asset_paths.extend(_string_list(required_assets.get(key)))

    local_assets_href = "/zh/docs/guides/local-assets" if locale == "zh" else "/docs/guides/local-assets"

    if locale == "zh":
        lines = [
            "### 运行环境",
            "",
            f"- **Conda 环境：** `{env_id}`。",
        ]
        if needs_new_env:
            lines.append("- **单独配置：** 需要 benchmark 专用 conda 环境，不要与统一 runtime 混用。")
        else:
            lines.append("- **单独配置：** 默认复用 WorldFoundry 统一环境（`needs_new_env: false`），通常不需要额外 benchmark-only conda env。")
        lines.extend(f"- {_escape_mdx(note)}" for note in profile_notes[:3])
        lines.extend(["", "### 测评数据", ""])
        if dataset_reason:
            lines.append(f"- {_escape_mdx(dataset_reason)}")
        if prompt_ref:
            lines.append(f"- Official prompt / task 资产：`{_escape_mdx(prompt_ref)}`。")
        if artifact_env:
            lines.append(
                f"- 待测模型输出目录：设置 `{artifact_env}` 指向生成视频或 rollout artifact 根目录。"
            )
        lines.extend(f"- `{path}`" for path in asset_paths[:4])
        lines.extend(["", "### Checkpoint 与资产", ""])
        if base_deps:
            lines.append(f"- Base-model / metric 依赖：{', '.join(f'`{dep}`' for dep in base_deps)}。")
        if checkpoint_refs:
            lines.extend(f"- `{ref}`" for ref in checkpoint_refs[:6])
        elif base_deps or preflight_notes:
            lines.append(f"- Metric / evaluator checkpoint 默认路径与覆盖变量见 [local assets 指南]({local_assets_href})。")
        else:
            lines.append("- 该 benchmark 以官方结果归一化或仿真 rollout 为主；policy / simulator checkpoint 见 catalog `requires` 与 task YAML。")
        lines.extend(f"- {_escape_mdx(note)}" for note in preflight_notes[:3])
        if required_env:
            lines.extend(["", "### 关键环境变量", ""])
            lines.extend(f"- `{item}`" for item in required_env[:8])
        if preflight:
            lines.extend(["", "### 环境检查", "", "```bash", preflight, "```"])
        return "\n".join(lines).rstrip()

    lines = [
        "### Runtime environment",
        "",
        f"- **Conda env:** `{env_id}`.",
    ]
    if needs_new_env:
        lines.append("- **Separate setup:** this benchmark expects a dedicated conda env rather than the unified runtime.")
    else:
        lines.append("- **Separate setup:** defaults to the unified WorldFoundry env (`needs_new_env: false`); a benchmark-only conda env is usually not required.")
    lines.extend(f"- {_escape_mdx(note)}" for note in profile_notes[:3])
    lines.extend(["", "### Evaluation data", ""])
    if dataset_reason:
        lines.append(f"- {_escape_mdx(dataset_reason)}")
    if prompt_ref:
        lines.append(f"- Official prompt/task assets: `{_escape_mdx(prompt_ref)}`.")
    if artifact_env:
        lines.append(
            f"- Candidate model outputs: set `{artifact_env}` to the generated-video or rollout artifact root."
        )
    lines.extend(f"- `{path}`" for path in asset_paths[:4])
    lines.extend(["", "### Checkpoints & assets", ""])
    if base_deps:
        lines.append(f"- Base-model / metric dependencies: {', '.join(f'`{dep}`' for dep in base_deps)}.")
    if checkpoint_refs:
        lines.extend(f"- `{ref}`" for ref in checkpoint_refs[:6])
    elif base_deps or preflight_notes:
        lines.append(f"- Default metric-checkpoint paths and override env vars are listed in the [local assets guide]({local_assets_href}).")
    else:
        lines.append("- This benchmark is primarily rollout- or official-result-driven; see catalog `requires` and the task YAML for policy/simulator checkpoints.")
    lines.extend(f"- {_escape_mdx(note)}" for note in preflight_notes[:3])
    if required_env:
        lines.extend(["", "### Key environment variables", ""])
        lines.extend(f"- `{item}`" for item in required_env[:8])
    if preflight:
        lines.extend(["", "### Verify setup", "", "```bash", preflight, "```"])
    return "\n".join(lines).rstrip()


def _build_environment_setup_section(catalog: Mapping[str, Any], benchmark_id: str, *, locale: str) -> str:
    body = _environment_setup_body(catalog, benchmark_id, locale=locale)
    if not body:
        return ""
    title = "环境准备" if locale == "zh" else "Environment setup"
    return f"## {title}\n\n{body}"


def _yaml_scalar(value: str) -> str:
    if any(ch in value for ch in ":{}[]#&*!|>'\"%@`"):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return value


def _render_page(catalog: Mapping[str, Any], benchmark_id: str, *, locale: str) -> str:
    name = str(catalog.get("name") or benchmark_id)
    description = (
        f"{name} — metrics, requirements, and run commands."
        if locale == "en"
        else f"{name} — 指标、依赖与运行命令。"
    )

    about_title = "About" if locale == "en" else "简介"
    setup_title_section = _build_environment_setup_section(catalog, benchmark_id, locale=locale)
    run_title = "Run Evaluation" if locale == "en" else "运行评测"
    artifacts_title = "Requirements" if locale == "en" else "依赖与产物"
    hub_label = "← Benchmark Hub" if locale == "en" else "← 返回 Benchmark Hub"
    hub_href = "/zh/docs/evaluation/benchmark-hub" if locale == "zh" else "/docs/evaluation/benchmark-hub"

    sections = [
        f"## {about_title}",
        "",
        _build_about_section(catalog, benchmark_id, locale=locale),
        "",
        _build_evaluates_section(catalog, benchmark_id, locale=locale),
        "",
    ]
    if setup_title_section:
        sections.extend([setup_title_section, ""])
    sections.extend(
        [
            f"## {run_title}",
            "",
            _build_run_section(catalog, benchmark_id, locale=locale),
            "",
            f"## {artifacts_title}",
            "",
            _build_artifacts_section(catalog, locale=locale),
            "",
            f"[{hub_label}]({hub_href})",
        ]
    )

    body = "\n\n".join(sections)

    return (
        f"---\ntitle: {_yaml_scalar(name)}\ndescription: {_yaml_scalar(description)}\n---\n\n{body}\n"
    )


def _ordered_benchmark_ids() -> list[str]:
    return [benchmark_id for _, _, ids in CATEGORY_GROUPS for benchmark_id in ids]


def _integration_status(catalog: Mapping[str, Any]) -> str:
    integration = catalog.get("integration")
    if isinstance(integration, Mapping):
        return str(integration.get("status") or "planned").strip().lower()
    return "planned"


def _verification_status(catalog: Mapping[str, Any]) -> str:
    runner = catalog.get("runner")
    if isinstance(runner, Mapping):
        return str(runner.get("verification_status") or "").strip().lower()
    return ""


def _public_badges(catalog: Mapping[str, Any]) -> list[str]:
    integration_status = _integration_status(catalog)
    verification_status = _verification_status(catalog)
    if verification_status == "normalizer_only":
        return ["normalizer"]
    if integration_status == "integrated":
        return ["integrated"]
    if integration_status == "blocked":
        return ["blocked"]
    if integration_status == "planned":
        return ["planned"]
    return []


def _write_catalog_status() -> None:
    payload: dict[str, dict[str, Any]] = {}
    for benchmark_id in _ordered_benchmark_ids():
        catalog_path = _catalog_path(benchmark_id)
        if catalog_path is None or not catalog_path.is_file():
            continue
        catalog = _load_yaml(catalog_path)
        category_en, category_zh = CATEGORY_BY_ID.get(benchmark_id, ("", ""))
        payload[benchmark_id] = {
            "name": str(catalog.get("name") or benchmark_id),
            "integrationStatus": _integration_status(catalog),
            "verificationStatus": _verification_status(catalog),
            "badges": _public_badges(catalog),
            "category": category_en,
            "categoryZh": category_zh,
        }
    CATALOG_STATUS_OUT.parent.mkdir(parents=True, exist_ok=True)
    CATALOG_STATUS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_meta() -> None:
    pages: list[str] = ["index"]
    for label_en, _label_zh, ids in CATEGORY_GROUPS:
        pages.append(f"---{label_en}---")
        pages.extend(ids)
    payload = {"title": "Benchmark Hub", "defaultOpen": True, "pages": pages}
    payload_zh = {"title": "Benchmark Hub", "defaultOpen": True, "pages": pages}
    (OUT_DIR / "meta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT_DIR / "meta.zh.json").write_text(json.dumps(payload_zh, indent=2) + "\n", encoding="utf-8")


def _append_index_links() -> None:
    for suffix, locale in (("", "en"), (".zh", "zh")):
        index_path = OUT_DIR / f"index{suffix}.mdx"
        if not index_path.is_file():
            continue
        hub_path = "/zh/docs/evaluation/benchmark-hub" if locale == "zh" else "/docs/evaluation/benchmark-hub"
        heading = "## Benchmarks" if locale == "zh" else "## Benchmarks"
        intro = (
            "各 benchmark 独立页面由 catalog manifest 与 task YAML 生成。"
            if locale == "zh"
            else "Per-benchmark pages are generated from catalog manifests and task YAML."
        )
        blocks: list[str] = [heading, "", intro, ""]
        for label_en, label_zh, ids in CATEGORY_GROUPS:
            label = label_zh if locale == "zh" else label_en
            blocks.append(f"### {label}")
            blocks.append("")
            for benchmark_id in ids:
                catalog_path = _catalog_path(benchmark_id)
                name = benchmark_id
                if catalog_path and catalog_path.is_file():
                    name = str(_load_yaml(catalog_path).get("name") or benchmark_id)
                blocks.append(f"- [{name}]({hub_path}/{benchmark_id})")
            blocks.append("")
        marker = "{/* benchmark-subpages */}"
        text = index_path.read_text(encoding="utf-8")
        if marker in text:
            prefix = text.split(marker, 1)[0].rstrip()
            index_path.write_text(prefix + "\n\n" + marker + "\n\n" + "\n".join(blocks).rstrip() + "\n", encoding="utf-8")
        else:
            index_path.write_text(text.rstrip() + "\n\n" + marker + "\n\n" + "\n".join(blocks).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    missing_catalog: list[str] = []
    for benchmark_id in _ordered_benchmark_ids():
        catalog_path = _catalog_path(benchmark_id)
        if catalog_path is None:
            missing_catalog.append(benchmark_id)
            continue
        catalog = _load_yaml(catalog_path)
        (OUT_DIR / f"{benchmark_id}.mdx").write_text(_render_page(catalog, benchmark_id, locale="en"), encoding="utf-8")
        (OUT_DIR / f"{benchmark_id}.zh.mdx").write_text(_render_page(catalog, benchmark_id, locale="zh"), encoding="utf-8")
        generated.append(benchmark_id)
    _write_meta()
    _write_catalog_status()
    _append_index_links()
    print(f"Generated {len(generated)} benchmark subpages in {OUT_DIR}")
    print(f"Wrote catalog status map to {CATALOG_STATUS_OUT}")
    if missing_catalog:
        raise SystemExit(f"Missing catalog manifests for: {', '.join(missing_catalog)}")


if __name__ == "__main__":
    main()
