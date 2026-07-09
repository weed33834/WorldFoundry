#!/usr/bin/env python3
"""Inline model-asset notes into metrics-usage partials and drop the standalone section."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs" / "fumadocs"
METRICS_EN = DOCS / "content/docs/evaluation/metrics.mdx"
METRICS_ZH = DOCS / "content/docs/evaluation/metrics.zh.mdx"
USAGE_EN = DOCS / "mdx/partials/metrics-usage.mdx"
USAGE_ZH = DOCS / "mdx/partials/metrics-usage.zh.mdx"

ASSETS_EN: dict[str, str] = {
    "artifact_count": "No model checkpoint — `GenerationResult` JSON/JSONL only.",
    "required_artifacts_present": "No model checkpoint — `GenerationResult` JSON/JSONL only.",
    "has_artifact": "No model checkpoint — `GenerationResult` JSON/JSONL only.",
    "numeric": "No model checkpoint — `GenerationResult` JSON/JSONL only.",
    "numeric_value": "No model checkpoint — `GenerationResult` JSON/JSONL only.",
    "clip_score": "OpenCLIP / BLIP2 / PickScore / InternVideo2 / UMT weights via `WORLDFOUNDRY_T2V_METRICS_CACHE_DIR` → `WORLDFOUNDRY_HFD_ROOT` → package `hf_cache`.",
    "vqa_score": "Hugging Face / LAVIS / WorldFoundry base-model caches for local VQA models; API models need provider credentials (`OPENAI_API_KEY`, Gemini key, etc.).",
    "itm_score": "Same cache order as CLIPScore. BLIP2/LAVIS URLs, InternVideo2 `internvl_c_13b_224px`, UMT `b16_ptk710_f8_res224` or `l16_ptk710_f8_res224`.",
    "facescore": "Hugging Face repo `AIGCer-OPPO/FaceScore` (default `~/.cache/FaceScore`); pass `model_name` as a local checkpoint path for offline runs.",
    "artscore": "Explicit ArtScore `.pt` checkpoint plus torchvision ResNet/VGG backbones. Stage under `WORLDFOUNDRY_CKPT_DIR` and pass `checkpoint_path=...`.",
    "facesim_cur": "`WORLDFOUNDRY_OPENS2V_WEIGHT_DIR` or `BestWishYsh/OpenS2V-Weight` with `face_extractor/` and `glint360k_curricular_face_r101_backbone.bin`.",
    "gme_score": "`WORLDFOUNDRY_GME_MODEL_PATH` or `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct/` Hugging Face snapshot.",
    "nexus_score": "`WORLDFOUNDRY_YOLO_WORLD_CKPT`, `WORLDFOUNDRY_YOLO_CLIP_MODEL` (`openai/clip-vit-base-patch32`), and `WORLDFOUNDRY_GME_MODEL_PATH` for offline runs.",
    "natural_score": "No local checkpoint — OpenAI-compatible judge API (`OPENAI_API_KEY` or `api_key=...`).",
    "fid": "Torch-fidelity InceptionV3 / CLIP / SwAV weights via `torch.hub` and `torchvision` caches. Offline: `feature_extractor_weights_path` or staged Torch cache.",
    "scene_fid": "Same torch-fidelity feature extractors as `fid`; bbox JSON drives crop extraction.",
    "inception_score": "Torch-fidelity InceptionV3 via `torch.hub` / `torchvision` cache.",
    "kid": "Torch-fidelity Inception features via `torch.hub` / `torchvision` cache.",
    "precision_recall": "Torch-fidelity Inception features via `torch.hub` / `torchvision` cache.",
    "improved_precision_recall": "Torchvision VGG16 feature extractor (`TORCH_HOME` / torch cache).",
    "realism_score": "Torchvision VGG16 feature extractor (`TORCH_HOME` / torch cache).",
    "fwd": "Torch-fidelity / wavelet backend weights via vendored distribution helpers and torch cache.",
    "cmmd": "CLIP vision encoder `openai/clip-vit-large-patch14-336` in Hugging Face `transformers` cache (`WORLDFOUNDRY_HFD_ROOT`).",
    "clean_fid": "Clean-FID Inception network and optional reference statistics in the `cleanfid` runtime cache.",
    "mind": "Torch-fidelity feature extractors via `torch.hub` / `torchvision` cache.",
    "trend": "No model checkpoint when caller supplies the required statistics/features.",
    "ppl": "Torch-fidelity latent / perceptual path weights via `torch.hub` / `torchvision` cache.",
    "vendi_score": "Image helper: torchvision InceptionV3. Text helper: `roberta-base` and `princeton-nlp/unsup-simcse-roberta-base` under `WORLDFOUNDRY_HFD_ROOT`. Feature-array mode needs no extractor.",
    "rke": "No model checkpoint when caller supplies feature embeddings.",
    "rnd": "No model checkpoint when caller supplies feature embeddings.",
    "rarity_score": "Feature bank from real images; uses the metric's configured feature extractor cache when built from media.",
    "fld": "No model checkpoint when caller supplies train/test/generated feature arrays.",
    "multimodal_mid": "No model checkpoint when caller supplies paired feature arrays.",
    "fjd": "No model checkpoint when caller supplies joint embeddings.",
    "crosslid": "No model checkpoint when caller supplies feature arrays.",
    "cfid": "No model checkpoint when caller supplies conditional feature arrays.",
    "ssd": "Feature extractor depends on API path; array mode needs no checkpoint.",
    "linear_separability": "No model checkpoint — pass the confusion matrix / separability inputs required by the API.",
    "fdd": "`WORLDFOUNDRY_FDD_DAE_CKPT` or auto-download via `gdown` (Google Drive id `1j7MVFWYfRNZLQ3uChe7TGQG8L1Oaf9Gt`).",
    "cis": "No model checkpoint when caller supplies class-probability batches.",
    "attribute_sad": "No model checkpoint — pass precomputed HCS histograms (`hcs_real`, `hcs_gen`, `text_list`).",
    "attribute_pad": "No model checkpoint — pass precomputed HCS histograms (`hcs_real`, `hcs_gen`, `text_list`).",
    "fvd": "Inception I3D `i3d_pretrained_400.pt` via explicit arg → `WORLDFOUNDRY_FVD_I3D_CKPT` → `WORLDFOUNDRY_MIRABENCH_FVD_I3D_CKPT` → `WORLDFOUNDRY_CKPT_DIR`.",
    "fvmd": "PIPs2 tracker `pips2_weights.pth` from FVMD release via torch hub cache.",
    "jedi": "Live path: V-JEPA dir (`WORLDFOUNDRY_JEDI_MODEL_DIR` / `WORLDFOUNDRY_JEDI_VJEPA_DIR` / `WORLDFOUNDRY_VJEPA_MODEL_DIR`, optional `WORLDFOUNDRY_JEDI_CONFIG_PATH`). Feature mode: precomputed train/test arrays only (`WORLDFOUNDRY_JEDI_FEATURE_PATH`).",
    "lpips": "VGG16 and LPIPS / DISTS / PieAPP auxiliary weights via torchvision and torch hub release URLs.",
    "ssim": "No model checkpoint — aligned image arrays only.",
    "ms_ssim": "No model checkpoint — aligned image arrays only.",
    "psnr": "No model checkpoint — aligned image arrays only.",
    "dino_similarity": "`facebook/dinov2-base` or `facebook/dino-vitb16` via base-model capability, then HF `from_pretrained`. Override with `WORLDFOUNDRY_DINOV2_BASE_MODEL_DIR` / `WORLDFOUNDRY_DINO_VITB16_MODEL_DIR`.",
    "dreamsim": "DreamSim release ZIPs from `ssundaram21/dreamsim`; pass `cache_dir=...` to `compute_dreamsim(...)`.",
    "fsim": "No model checkpoint — aligned image arrays only.",
    "cpbd": "No model checkpoint — single image input only.",
    "mask_accuracy": "No model checkpoint — predicted and ground-truth masks only.",
    "object_detection": "No model checkpoint — phrase/box inputs per WorldScore-style protocol.",
    "lqs": "No model checkpoint — predicted and ground-truth layout boxes only.",
    "semsr": "CLIP backbone weights via scorer / HF cache when computing from images.",
    "irs": "No model checkpoint — image statistics computed from inputs.",
    "cas": "No fixed checkpoint — trains a lightweight classifier on synthetic features supplied by the API.",
    "manipulation_direction": "CLIP image/text encoders via HF or scorer cache.",
    "vs_similarity": "HDGAN embedding inputs supplied by the API; no bundled standalone checkpoint.",
    "quality_loss": "CLIPScore-style CLIP weights via `WORLDFOUNDRY_T2V_METRICS_CACHE_DIR` / HF cache.",
    "object_wise_consistency": "No model checkpoint — guidance boxes and detector outputs only.",
}

ASSETS_ZH: dict[str, str] = {
    "artifact_count": "无需模型 checkpoint — 只需要 `GenerationResult` JSON/JSONL。",
    "required_artifacts_present": "无需模型 checkpoint — 只需要 `GenerationResult` JSON/JSONL。",
    "has_artifact": "无需模型 checkpoint — 只需要 `GenerationResult` JSON/JSONL。",
    "numeric": "无需模型 checkpoint — 只需要 `GenerationResult` JSON/JSONL。",
    "numeric_value": "无需模型 checkpoint — 只需要 `GenerationResult` JSON/JSONL。",
    "clip_score": "OpenCLIP / BLIP2 / PickScore / InternVideo2 / UMT 权重，cache 顺序为 `WORLDFOUNDRY_T2V_METRICS_CACHE_DIR` → `WORLDFOUNDRY_HFD_ROOT` → package `hf_cache`。",
    "vqa_score": "本地 VQA 模型走 Hugging Face / LAVIS / WorldFoundry base-model cache；API 模型需要 provider 凭据（`OPENAI_API_KEY`、Gemini key 等）。",
    "itm_score": "与 CLIPScore 相同的 cache 顺序。BLIP2/LAVIS URL、InternVideo2 `internvl_c_13b_224px`、UMT `b16_ptk710_f8_res224` 或 `l16_ptk710_f8_res224`。",
    "facescore": "Hugging Face repo `AIGCer-OPPO/FaceScore`（默认 `~/.cache/FaceScore`）；离线可将 `model_name` 设为本地 checkpoint 路径。",
    "artscore": "需要显式 ArtScore `.pt` checkpoint，外加 torchvision ResNet/VGG backbone。放在 `WORLDFOUNDRY_CKPT_DIR` 并传 `checkpoint_path=...`。",
    "facesim_cur": "`WORLDFOUNDRY_OPENS2V_WEIGHT_DIR` 或 `BestWishYsh/OpenS2V-Weight`，目录内需有 `face_extractor/` 和 `glint360k_curricular_face_r101_backbone.bin`。",
    "gme_score": "`WORLDFOUNDRY_GME_MODEL_PATH` 或 Hugging Face snapshot `Alibaba-NLP/gme-Qwen2-VL-7B-Instruct/`。",
    "nexus_score": "离线需同时设置 `WORLDFOUNDRY_YOLO_WORLD_CKPT`、`WORLDFOUNDRY_YOLO_CLIP_MODEL`（`openai/clip-vit-base-patch32`）和 `WORLDFOUNDRY_GME_MODEL_PATH`。",
    "natural_score": "无本地 checkpoint — OpenAI-compatible judge API（`OPENAI_API_KEY` 或 `api_key=...`）。",
    "fid": "Torch-fidelity InceptionV3 / CLIP / SwAV 权重，经 `torch.hub` 与 `torchvision` cache 加载。离线可传 `feature_extractor_weights_path` 或提前放入 Torch cache。",
    "scene_fid": "与 `fid` 相同的 torch-fidelity feature extractor；bbox JSON 驱动 crop。",
    "inception_score": "Torch-fidelity InceptionV3，经 `torch.hub` / `torchvision` cache 加载。",
    "kid": "Torch-fidelity Inception feature，经 `torch.hub` / `torchvision` cache 加载。",
    "precision_recall": "Torch-fidelity Inception feature，经 `torch.hub` / `torchvision` cache 加载。",
    "improved_precision_recall": "Torchvision VGG16 feature extractor（`TORCH_HOME` / torch cache）。",
    "realism_score": "Torchvision VGG16 feature extractor（`TORCH_HOME` / torch cache）。",
    "fwd": "Vendored distribution helper / wavelet 后端权重，经 torch cache 加载。",
    "cmmd": "CLIP vision encoder `openai/clip-vit-large-patch14-336`，Hugging Face `transformers` cache（`WORLDFOUNDRY_HFD_ROOT`）。",
    "clean_fid": "Clean-FID Inception 网络与可选 reference statistics，放在 `cleanfid` runtime cache。",
    "mind": "Torch-fidelity feature extractor，经 `torch.hub` / `torchvision` cache 加载。",
    "trend": "调用方提供 API 要求的统计量或 feature 时，无需模型 checkpoint。",
    "ppl": "Torch-fidelity latent / perceptual path 权重，经 `torch.hub` / `torchvision` cache 加载。",
    "vendi_score": "图像 helper：torchvision InceptionV3。文本 helper：`roberta-base` 与 `princeton-nlp/unsup-simcse-roberta-base`（`WORLDFOUNDRY_HFD_ROOT`）。feature 数组模式无需 extractor。",
    "rke": "调用方提供 feature embedding 时，无需模型 checkpoint。",
    "rnd": "调用方提供 feature embedding 时，无需模型 checkpoint。",
    "rarity_score": "由真实图像构建 feature bank；从媒体抽取时会使用 metric 配置的 feature extractor cache。",
    "fld": "调用方提供 train/test/generated feature 数组时，无需模型 checkpoint。",
    "multimodal_mid": "调用方提供 paired feature 数组时，无需模型 checkpoint。",
    "fjd": "调用方提供 joint embedding 时，无需模型 checkpoint。",
    "crosslid": "调用方提供 feature 数组时，无需模型 checkpoint。",
    "cfid": "调用方提供 conditional feature 数组时，无需模型 checkpoint。",
    "ssd": "取决于 API 路径；数组模式无需 checkpoint。",
    "linear_separability": "无需模型 checkpoint — 传入 API 要求的 confusion matrix / separability 输入。",
    "fdd": "`WORLDFOUNDRY_FDD_DAE_CKPT`，或通过 `gdown` 自动下载（Google Drive id `1j7MVFWYfRNZLQ3uChe7TGQG8L1Oaf9Gt`）。",
    "cis": "调用方提供 class-probability batch 时，无需模型 checkpoint。",
    "attribute_sad": "无需模型 checkpoint — 传入预计算 HCS histogram（`hcs_real`、`hcs_gen`、`text_list`）。",
    "attribute_pad": "无需模型 checkpoint — 传入预计算 HCS histogram（`hcs_real`、`hcs_gen`、`text_list`）。",
    "fvd": "Inception I3D `i3d_pretrained_400.pt`：显式参数 → `WORLDFOUNDRY_FVD_I3D_CKPT` → `WORLDFOUNDRY_MIRABENCH_FVD_I3D_CKPT` → `WORLDFOUNDRY_CKPT_DIR`。",
    "fvmd": "FVMD release 的 PIPs2 tracker `pips2_weights.pth`，经 torch hub cache 加载。",
    "jedi": "实时路径：V-JEPA 目录（`WORLDFOUNDRY_JEDI_MODEL_DIR` / `WORLDFOUNDRY_JEDI_VJEPA_DIR` / `WORLDFOUNDRY_VJEPA_MODEL_DIR`，可选 `WORLDFOUNDRY_JEDI_CONFIG_PATH`）。feature 模式：仅预计算 train/test 数组（`WORLDFOUNDRY_JEDI_FEATURE_PATH`）。",
    "lpips": "VGG16 与 LPIPS / DISTS / PieAPP 辅助权重，经 torchvision 与 torch hub release URL 加载。",
    "ssim": "无需模型 checkpoint — 只需要对齐图像数组。",
    "ms_ssim": "无需模型 checkpoint — 只需要对齐图像数组。",
    "psnr": "无需模型 checkpoint — 只需要对齐图像数组。",
    "dino_similarity": "`facebook/dinov2-base` 或 `facebook/dino-vitb16`，先查 base-model capability 再走 HF `from_pretrained`。可用 `WORLDFOUNDRY_DINOV2_BASE_MODEL_DIR` / `WORLDFOUNDRY_DINO_VITB16_MODEL_DIR` 覆盖。",
    "dreamsim": "`ssundaram21/dreamsim` release ZIP；向 `compute_dreamsim(..., cache_dir=...)` 传入本地目录。",
    "fsim": "无需模型 checkpoint — 只需要对齐图像数组。",
    "cpbd": "无需模型 checkpoint — 只需要单张图像输入。",
    "mask_accuracy": "无需模型 checkpoint — 只需要预测 mask 与 GT mask。",
    "object_detection": "无需模型 checkpoint — 按 WorldScore 协议提供 phrase/box 输入。",
    "lqs": "无需模型 checkpoint — 只需要预测与 GT layout bbox。",
    "semsr": "从图像计算时使用 CLIP backbone 权重（scorer / HF cache）。",
    "irs": "无需模型 checkpoint — 由输入图像直接计算统计量。",
    "cas": "无固定 checkpoint — 在 API 提供的 synthetic feature 上训练轻量分类器。",
    "manipulation_direction": "CLIP 图像/文本 encoder，经 HF 或 scorer cache 加载。",
    "vs_similarity": "由 API 提供 HDGAN embedding 输入；无 bundled standalone checkpoint。",
    "quality_loss": "CLIPScore 风格 CLIP 权重，经 `WORLDFOUNDRY_T2V_METRICS_CACHE_DIR` / HF cache 加载。",
    "object_wise_consistency": "无需模型 checkpoint — 只需要 guidance box 与 detector 输出。",
}

ENV_BLOCK_EN = """```bash
export WORLDFOUNDRY_HFD_ROOT=${HF_HOME:-$HOME/.cache/huggingface}
export WORLDFOUNDRY_T2V_METRICS_CACHE_DIR=$WORLDFOUNDRY_HFD_ROOT
export WORLDFOUNDRY_CKPT_DIR=$HOME/.cache/worldfoundry/checkpoints
```"""

ENV_BLOCK_ZH = """```bash
export WORLDFOUNDRY_HFD_ROOT=${HF_HOME:-$HOME/.cache/huggingface}
export WORLDFOUNDRY_T2V_METRICS_CACHE_DIR=$WORLDFOUNDRY_HFD_ROOT
export WORLDFOUNDRY_CKPT_DIR=$HOME/.cache/worldfoundry/checkpoints
```"""

ASSETS_HEADING_EN = "## Model Assets And Checkpoints"
ASSETS_HEADING_ZH = "## 模型资产与 checkpoint"


def _metric_ids_from_heading(heading: str) -> list[str]:
    text = heading.strip()
    text = re.sub(r"^###\s+", "", text)
    text = re.sub(r"\s*\([↑↓]\)\s*$", "", text)
    text = text.strip("`").strip()
    ids: list[str] = []
    for part in text.split("/"):
        token = part.strip().strip("`")
        if not token:
            continue
        base = token.split(":", 1)[0].strip()
        if base == "numeric":
            ids.extend(["numeric", "numeric_value"])
        elif base == "has_artifact":
            ids.append("has_artifact")
        else:
            ids.append(base)
    return list(dict.fromkeys(ids))


def _assets_block(metric_ids: list[str], assets: dict[str, str], *, locale: str) -> str:
    lines: list[str] = []
    for metric_id in metric_ids:
        text = assets.get(metric_id)
        if not text:
            continue
        if len(metric_ids) == 1:
            lines.append(text)
        else:
            lines.append(f"- **`{metric_id}`:** {text}")
    if not lines:
        return ""
    header = "**Assets:**" if locale == "en" else "**资产：**"
    if len(lines) == 1:
        return f"{header} {lines[0]}"
    return header + "\n" + "\n".join(lines)


def _remove_assets_blocks(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    prefixes = ("**Assets:**", "**资产：**", "**资产:**")
    while i < len(lines):
        line = lines[i]
        if any(line.startswith(prefix) for prefix in prefixes):
            i += 1
            while i < len(lines):
                if lines[i].strip() == "":
                    i += 1
                    continue
                if not lines[i].startswith("- **`"):
                    break
                if any(token in lines[i] for token in ("Paper:", "Repo:", "论文:", "仓库:")):
                    break
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def _inline_assets(usage_text: str, assets: dict[str, str], *, locale: str) -> str:
    ref_markers = ("**Reference:**", "**参考：**", "**参考:**")
    asset_prefixes = ("**Assets:**", "**资产：**", "**资产:**")
    out: list[str] = []
    lines = usage_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("### ") and not line.startswith("### Run multiple"):
            metric_ids = _metric_ids_from_heading(line)
            out.append(line)
            i += 1
            inserted = False
            while i < len(lines) and not lines[i].startswith("### ") and not lines[i].startswith("## "):
                current = lines[i]
                if any(current.startswith(prefix) for prefix in asset_prefixes):
                    i += 1
                    while i < len(lines) and (lines[i].startswith("- **`") or lines[i].strip() == ""):
                        i += 1
                    continue
                if not inserted and any(current.startswith(marker) for marker in ref_markers):
                    ref_lines = [current]
                    i += 1
                    while i < len(lines) and lines[i].startswith("- **`"):
                        ref_lines.append(lines[i])
                        i += 1
                    out.extend(ref_lines)
                    block = _assets_block(metric_ids, assets, locale=locale)
                    if block:
                        out.append("")
                        out.extend(block.splitlines())
                    inserted = True
                    continue
                out.append(current)
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).rstrip() + "\n"


def _strip_assets_section(text: str, heading: str) -> str:
    if heading not in text:
        return text
    before, rest = text.split(heading, 1)
    next_heading = re.search(r"\n## ", rest)
    after = rest[next_heading.start() :] if next_heading else ""
    return before.rstrip() + "\n\n" + after.lstrip("\n")


def _ensure_env_note(text: str, *, locale: str) -> str:
    if "WORLDFOUNDRY_CKPT_DIR=$HOME/.cache/worldfoundry/checkpoints" in text:
        return text
    note_en = (
        "Metric imports are lazy: checkpoints are required only when a metric loads a feature extractor, "
        "multimodal scorer, or API judge. Default caches follow Hugging Face / Torch / torchvision layout:\n\n"
        f"{ENV_BLOCK_EN}\n"
    )
    note_zh = (
        "Metric 模块保持 lazy import：只有实际加载 feature extractor、多模态 scorer 或 API judge 时才需要 checkpoint。"
        "默认 cache 沿用 Hugging Face / Torch / torchvision layout：\n\n"
        f"{ENV_BLOCK_ZH}\n"
    )
    needle = "<MetricQuickNav locale=\"en\" />" if locale == "en" else "<MetricQuickNav locale=\"zh\" />"
    insert = note_en if locale == "en" else note_zh
    if needle in text:
        return text.replace(needle, f"{needle}\n\n{insert}", 1)
    return text


def main() -> None:
    metrics_en = METRICS_EN.read_text(encoding="utf-8")
    metrics_zh = METRICS_ZH.read_text(encoding="utf-8")

    for usage_path, assets, locale in ((USAGE_EN, ASSETS_EN, "en"), (USAGE_ZH, ASSETS_ZH, "zh")):
        cleaned = _remove_assets_blocks(usage_path.read_text(encoding="utf-8"))
        updated = _inline_assets(cleaned, assets, locale=locale)
        usage_path.write_text(_ensure_env_note(updated, locale=locale), encoding="utf-8")

    METRICS_EN.write_text(_strip_assets_section(metrics_en, ASSETS_HEADING_EN), encoding="utf-8")
    METRICS_ZH.write_text(_strip_assets_section(metrics_zh, ASSETS_HEADING_ZH), encoding="utf-8")

    print(f"Inlined assets into {USAGE_EN.name} and {USAGE_ZH.name}")


if __name__ == "__main__":
    main()
