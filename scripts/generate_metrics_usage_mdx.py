#!/usr/bin/env python3
"""Generate per-metric usage MDX fragments for docs/fumadocs."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_EN = ROOT / "docs/fumadocs/mdx/partials/metrics-usage.mdx"
OUT_ZH = ROOT / "docs/fumadocs/mdx/partials/metrics-usage.zh.mdx"


def recipe(
    metric_id: str,
    *,
    direction: str,
    when_en: str,
    when_zh: str,
    code: str,
    returns_en: str,
    returns_zh: str,
    extra_en: str = "",
    extra_zh: str = "",
) -> dict:
    return {
        "id": metric_id,
        "direction": direction,
        "when_en": when_en,
        "when_zh": when_zh,
        "code": code.strip(),
        "returns_en": returns_en,
        "returns_zh": returns_zh,
        "extra_en": extra_en.strip(),
        "extra_zh": extra_zh.strip(),
    }


RECIPES: list[dict] = [
    recipe(
        "artifact_count",
        direction="↑",
        when_en="You already ran inference and have a `GenerationResult` JSONL.",
        when_zh="已完成推理，手上有 `GenerationResult` JSONL。",
        code="""
worldfoundry-eval evaluate \\
  --mode existing-results \\
  --results-path runs/results.jsonl \\
  --output-dir runs/eval/artifact_count \\
  --metric artifact_count \\
  --json
""",
        returns_en="Scorecard JSON under `--output-dir` with per-row artifact counts.",
        returns_zh="`--output-dir` 下 scorecard JSON，含每行 artifact 计数。",
    ),
    recipe(
        "required_artifacts_present",
        direction="↑",
        when_en="Check that every result contains required output artifacts.",
        when_zh="检查每条 result 是否包含所有 required artifact。",
        code="""
worldfoundry-eval evaluate \\
  --mode existing-results \\
  --results-path runs/results.jsonl \\
  --output-dir runs/eval/required \\
  --required-artifact generated_artifact \\
  --required-artifact video \\
  --metric required_artifacts_present \\
  --json
""",
        returns_en="`1` when all `--required-artifact` names exist on a row, else `0`.",
        returns_zh="全部 `--required-artifact` 存在为 `1`，否则 `0`。",
    ),
    recipe(
        "has_artifact:<name>",
        direction="↑",
        when_en="Parameterized presence check for one artifact name.",
        when_zh="检查单个 artifact 是否存在（参数化 id）。",
        code="""
worldfoundry-eval evaluate \\
  --mode existing-results \\
  --results-path runs/results.jsonl \\
  --output-dir runs/eval/has_video \\
  --metric has_artifact:video \\
  --json
""",
        returns_en="`1` if artifact `video` exists on the row.",
        returns_zh="该行存在 `video` artifact 时为 `1`。",
    ),
    recipe(
        "numeric / numeric:<name>",
        direction="↑",
        when_en="Emit numeric fields already stored on each result row.",
        when_zh="输出 result 行上已有的 numeric 字段。",
        code="""
# all numeric fields
worldfoundry-eval evaluate ... --metric numeric

# one field, e.g. reward from model metadata
worldfoundry-eval evaluate ... --metric numeric:reward
""",
        returns_en="Named scalar(s) copied from result metadata/scores.",
        returns_zh="从 result metadata/scores 复制出的标量。",
    ),
    recipe(
        "clip_score",
        direction="↑",
        when_en="Score image–text alignment with CLIPScore.",
        when_zh="用 CLIPScore 评估图像–文本对齐。",
        code="""
from worldfoundry.evaluation.tasks.metrics import CLIPScore, list_all_clipscore_models

print(list_all_clipscore_models()[:5])

clip = CLIPScore(model="openai:ViT-L/14", device="cuda")
score = clip.forward(images="sample.png", texts="a red sports car on a wet street")
print(float(score.item()))
""",
        returns_en="Float tensor; higher = better alignment.",
        returns_zh="浮点张量；越高对齐越好。",
    ),
    recipe(
        "vqa_score",
        direction="↑",
        when_en="Probability a VQA model answers yes to a text probe about the image.",
        when_zh="VQA 模型对图像–文本探针回答「是」的概率。",
        code="""
from worldfoundry.evaluation.tasks.metrics import VQAScore

vqa = VQAScore(model="clip-flant5-xxl", device="cuda")
score = vqa.forward(
    images="sample.png",
    texts="Is there a dog wearing sunglasses in this image?",
)
print(float(score.item()))
""",
        returns_en="Probability in `[0, 1]`; higher = better.",
        returns_zh="`[0, 1]` 概率；越高越好。",
    ),
    recipe(
        "itm_score",
        direction="↑",
        when_en="Image–text matching (ITM) for images or videos.",
        when_zh="图像/视频的 image–text matching（ITM）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import ITMScore

itm = ITMScore(model="blip2-itm", device="cuda")
score = itm.forward(images="sample.png", texts="a cat sitting on a windowsill")
print(float(score.item()))
""",
        returns_en="Matching score tensor; higher = better match.",
        returns_zh="匹配分数张量；越高越好。",
    ),
    recipe(
        "facescore",
        direction="↑",
        when_en="Face quality reward on a single portrait image.",
        when_zh="单张人像的脸部质量 reward。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_facescore

score = compute_facescore("outputs/face.png", device="cuda")
print(score)
""",
        returns_en="Float reward; higher = better face quality.",
        returns_zh="浮点 reward；越高人脸质量越好。",
    ),
    recipe(
        "artscore",
        direction="↑",
        when_en="Estimate artness (fine-art vs photorealistic).",
        when_zh="估计艺术感（油画 vs 照片）。",
        code="""
from PIL import Image
from worldfoundry.evaluation.tasks.metrics import load_artscore_model

model = load_artscore_model(device="cuda")
image = Image.open("sample.png").convert("RGB")
score = model(image)
print(float(score))
""",
        returns_en="Artness score; higher = more art-like.",
        returns_zh="艺术感分数；越高越偏艺术风格。",
    ),
    recipe(
        "facesim_cur",
        direction="↑",
        when_en="Face cosine similarity between reference and generated portraits (OpenS2V).",
        when_zh="参考/生成人像间的 InsightFace+CurricularFace 余弦相似度。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_facesim_cur

score = compute_facesim_cur(
    ref_paths=["ref/face_a.png", "ref/face_b.png"],
    gen_paths=["gen/face_a.png", "gen/face_b.png"],
    device="cuda",
)
print(score)
""",
        returns_en="Mean face similarity; set `WORLDFOUNDRY_OPENS2V_WEIGHT_DIR` first.",
        returns_zh="平均人脸相似度；需先设置 `WORLDFOUNDRY_OPENS2V_WEIGHT_DIR`。",
    ),
    recipe(
        "gme_score / nexus_score / natural_score",
        direction="↑",
        when_en="OpenS2V subject-to-video evaluation with folder layout + JSON manifest.",
        when_zh="OpenS2V subject-to-video 评估（目录 + JSON manifest）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import (
    compute_gme_score,
    compute_nexus_score,
    compute_natural_score,
)

manifest = "opens2v/eval.json"  # keyed by video stem
videos = "opens2v/videos"
refs = "opens2v/ref_images"

gme = compute_gme_score(
    input_video_folder=videos,
    input_image_folder=refs,
    input_json_file=manifest,
    device="cuda",
)
nexus = compute_nexus_score(
    input_video_folder=videos,
    input_image_folder=refs,
    input_json_file=manifest,
    device="cuda",
)
natural = compute_natural_score(
    input_video_folder=videos,
    input_json_file=manifest,
)  # needs OPENAI_API_KEY
print(gme, nexus, natural)
""",
        returns_en="Dict payloads per video; see OpenS2V eval JSON schema.",
        returns_zh="按视频的 dict 结果；见 OpenS2V eval JSON 结构。",
        extra_en="Manifest entry example: `{\"video_001\": {\"img_paths\": [...], \"class_label\": [...]}}`.",
        extra_zh="Manifest 示例：`{\"video_001\": {\"img_paths\": [...], \"class_label\": [...]}}`。",
    ),
    recipe(
        "fid",
        direction="↓",
        when_en="Compare two folders of images (reference vs generated).",
        when_zh="比较两个图像文件夹（reference vs generated）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_fid

ref_dir = "/data/coco_val"
gen_dir = "/data/my_model_samples"

fid = compute_fid(ref_dir, gen_dir)
print(fid)  # lower is better

clip_fid = compute_fid(ref_dir, gen_dir, feature_extractor="clip-vit-b-32")
swav_fid = compute_fid(ref_dir, gen_dir, feature_extractor="swav-resnet50")
""",
        returns_en="Float FID; lower is better.",
        returns_zh="浮点 FID；越低越好。",
    ),
    recipe(
        "scene_fid",
        direction="↓",
        when_en="SceneFID with object crops from bbox annotations.",
        when_zh="带 bbox 物体裁剪的 SceneFID。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_scene_fid

score = compute_scene_fid(
    "/data/ref",
    "/data/gen",
    reference_bboxes_json="/data/ref_bboxes.json",
)
print(score)
""",
        returns_en="Scene-level FID float.",
        returns_zh="Scene 级 FID 浮点数。",
    ),
    recipe(
        "inception_score",
        direction="↑",
        when_en="Measure generated-set quality/diversity via Inception classifier.",
        when_zh="用 Inception 分类器衡量生成集质量/多样性。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_inception_score

result = compute_inception_score("/data/generated_only")
print(result)  # dict with mean/std IS
""",
        returns_en="Dict with Inception Score mean/std; higher mean is better.",
        returns_zh="含 IS mean/std 的 dict；mean 越高越好。",
    ),
    recipe(
        "kid",
        direction="↓",
        when_en="Kernel Inception Distance between two image sets.",
        when_zh="两图像集合间的 Kernel Inception Distance。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_kid

kid = compute_kid("/data/ref", "/data/gen")
print(kid["kernel_inception_distance_mean"])
""",
        returns_en="Dict of KID statistics; lower mean is better.",
        returns_zh="KID 统计 dict；mean 越低越好。",
    ),
    recipe(
        "precision_recall",
        direction="↑",
        when_en="Manifold precision/recall for coverage and fidelity.",
        when_zh="流形 precision/recall（覆盖度与保真度）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_precision_recall

pr = compute_precision_recall("/data/ref", "/data/gen")
print(pr)
""",
        returns_en="Dict with precision, recall, and related keys.",
        returns_zh="含 precision、recall 等键的 dict。",
    ),
    recipe(
        "improved_precision_recall",
        direction="↑",
        when_en="α-precision, β-recall, realism score (k-NN radii).",
        when_zh="α-precision、β-recall、realism score（k-NN 半径）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import (
    compute_improved_precision_recall,
    compute_realism_score,
)

ipr = compute_improved_precision_recall("/data/ref", "/data/gen")
realism = compute_realism_score("/data/ref", "/data/gen")
print(ipr, realism)
""",
        returns_en="Dict of IPR metrics; higher precision/recall/realism is better.",
        returns_zh="IPR 指标 dict；precision/recall/realism 越高越好。",
    ),
    recipe(
        "fwd / cmmd / clean_fid / mind / trend",
        direction="↓",
        when_en="Distribution distance on image folders (different feature backends).",
        when_zh="图像文件夹上的分布距离（不同特征后端）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import (
    compute_fwd,
    compute_cmmd,
    compute_clean_fid,
    compute_mind,
    compute_trend,
)

ref, gen = "/data/ref", "/data/gen"
print("fwd", compute_fwd(ref, gen))
print("cmmd", compute_cmmd(ref, gen))
print("clean_fid", compute_clean_fid(ref, gen, dataset_name="cifar10", mode="clean"))
print("mind", compute_mind(ref, gen))
print("trend", compute_trend(ref, gen))
""",
        returns_en="Float or dict per metric; lower is better for these ids.",
        returns_zh="各 metric 返回 float 或 dict；这些 id 越低越好。",
    ),
    recipe(
        "ppl",
        direction="↓",
        when_en="StyleGAN latent perceptual path length (smoothness).",
        when_zh="StyleGAN 潜空间感知路径长度（平滑度）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_ppl

ppl = compute_ppl("/path/to/generated_or_latent_inputs")
print(ppl)
""",
        returns_en="PPL statistic; lower = smoother generator.",
        returns_zh="PPL 统计量；越低生成器越平滑。",
    ),
    recipe(
        "vendi_score / rke / rnd",
        direction="↑",
        when_en="Diversity from feature matrices or image folders.",
        when_zh="由特征矩阵或图像目录计算多样性。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import (
    compute_vendi_score,
    compute_rke,
    compute_rrke,
    compute_rnd,
    compute_rnd_from_images,
)

features = np.load("embeddings.npy")  # shape (N, D)
print("vendi", compute_vendi_score(features))
print("rke", compute_rke(features))
print("rrke", compute_rrke(ref_features, gen_features))
print("rnd", compute_rnd(features))
print("rnd_images", compute_rnd_from_images("/data/generated"))
""",
        returns_en="Float diversity scores; higher vendi/rke/rnd = more diverse.",
        returns_zh="多样性浮点分数；vendi/rke/rnd 越高越多样。",
    ),
    recipe(
        "rarity_score",
        direction="↑",
        when_en="How rare generated samples are vs a real feature bank.",
        when_zh="生成样本相对真实特征库的罕见程度。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import compute_mean_rarity_score

real_feats = np.load("real.npy")
gen_feats = np.load("gen.npy")
print(compute_mean_rarity_score(real_feats, gen_feats, k=5))
""",
        returns_en="Mean rarity; higher = rarer samples.",
        returns_zh="平均 rarity；越高越罕见。",
    ),
    recipe(
        "fld / multimodal_mid / fjd / crosslid / cfid / ssd",
        direction="mixed",
        when_en="Feature-array metrics for fidelity, multimodal alignment, or diversity.",
        when_zh="基于特征数组的保真度、多模态对齐或多样性指标。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import (
    compute_fld,
    compute_multimodal_mid,
    compute_fjd_from_joint_embeddings,
    compute_crosslid,
    compute_cfid,
    compute_ssd,
)

train = np.load("train_feats.npy")
test = np.load("test_feats.npy")
gen = np.load("gen_feats.npy")
print("fld", compute_fld(train, test, gen))
print("mid", compute_multimodal_mid(image_feats, text_feats))
print("fjd", compute_fjd_from_joint_embeddings(real_joint, gen_joint))
print("crosslid", compute_crosslid(features, k=5))
print("cfid", compute_cfid(ref_pairs, gen_pairs))
print("ssd", compute_ssd(text_embeds, image_embeds))
""",
        returns_en="Float or dict; check each function docstring for direction.",
        returns_zh="float 或 dict；方向见各函数 docstring。",
    ),
    recipe(
        "linear_separability",
        direction="↑",
        when_en="StyleGAN latent linear separability from an SVM confusion matrix.",
        when_zh="由 StyleGAN 潜空间 SVM 混淆矩阵计算线性可分性。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import compute_linear_separability

confusion = np.load("svm_confusion.npy")
print(compute_linear_separability(confusion))
""",
        returns_en="Separability score; higher = more disentangled latents.",
        returns_zh="可分性分数；越高潜变量越解耦。",
    ),
    recipe(
        "fdd",
        direction="↓",
        when_en="Fréchet Denoised Distance on DAE latent features.",
        when_zh="DAE 潜特征上的 Fréchet Denoised Distance。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_fdd

# export WORLDFOUNDRY_FDD_DAE_CKPT=/path/to/dae.pt
score = compute_fdd("/data/ref", "/data/gen")
print(score)
""",
        returns_en="Float FDD; lower = more structurally plausible.",
        returns_zh="FDD 浮点；越低结构越合理。",
    ),
    recipe(
        "cis",
        direction="↑",
        when_en="Conditional Inception Score from class-probability batches.",
        when_zh="由类概率 batch 计算 Conditional Inception Score。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import compute_cis, compute_cis_from_predictions

# probs: list of (N, num_classes) arrays, one bucket per condition
print(compute_cis(class_probs))
print(compute_cis_from_predictions(predictions, num_classes=1000))
""",
        returns_en="Dict with BCIS, WCIS, and combined CIS.",
        returns_zh="含 BCIS、WCIS 与组合 CIS 的 dict。",
    ),
    recipe(
        "attribute_sad / attribute_pad",
        direction="↓",
        when_en="SadPaD attribute divergence from precomputed HCS histograms.",
        when_zh="由预计算 HCS 直方图计算 SadPaD 属性散度。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_attribute_sad, compute_attribute_pad

text_list = ["smiling", "young", "eyeglasses"]
sad = compute_attribute_sad(hcs_real, hcs_gen, text_list)
pad = compute_attribute_pad(hcs_real, hcs_gen, text_list)
print(sad, pad)
""",
        returns_en="Dict with per-attribute divergences; lower = closer to real distribution.",
        returns_zh="含各属性散度的 dict；越低越接近真实分布。",
    ),
    recipe(
        "fvd",
        direction="↓",
        when_en="Fréchet Video Distance between real and generated video sets.",
        when_zh="真实/生成视频集合间的 Fréchet Video Distance。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import (
    compute_fvd_from_numpy,
    compute_fvd_from_frame_dirs,
)

# uint8 arrays shaped (N, T, H, W, C)
real = np.load("real_videos.npy")
gen = np.load("gen_videos.npy")
print(compute_fvd_from_numpy(real, gen, device="cuda"))

print(
    compute_fvd_from_frame_dirs(
        reference_frame_dirs=["/data/ref_frames/vid0"],
        generated_frame_dirs=["/data/gen_frames/vid0"],
        device="cuda",
    )
)
""",
        returns_en="Float FVD; set `WORLDFOUNDRY_FVD_I3D_CKPT` if needed.",
        returns_zh="FVD 浮点；必要时设置 `WORLDFOUNDRY_FVD_I3D_CKPT`。",
    ),
    recipe(
        "fvmd",
        direction="↓",
        when_en="Motion-feature Fréchet distance between video directories.",
        when_zh="视频目录间基于运动特征的 Fréchet 距离。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_fvmd

score = compute_fvmd("/data/ref_videos", "/data/gen_videos")
print(score)
""",
        returns_en="Float FVMD; lower = closer motion distribution.",
        returns_zh="FVMD 浮点；越低运动分布越接近。",
    ),
    recipe(
        "jedi",
        direction="↓",
        when_en="Video distribution quality via V-JEPA feature MMD.",
        when_zh="V-JEPA 特征 MMD 衡量视频分布质量。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import (
    compute_jedi_from_features,
    compute_mock_jedi,
)

train_feats = np.load("jedi_train.npy")
test_feats = np.load("jedi_test.npy")
print(compute_jedi_from_features(train_feats, test_feats))

# smoke test without checkpoints
print(compute_mock_jedi())
""",
        returns_en="Float JEDi MMD; configure `WORLDFOUNDRY_JEDI_*` for live backend.",
        returns_zh="JEDi MMD 浮点；live backend 需配置 `WORLDFOUNDRY_JEDI_*`。",
    ),
    recipe(
        "lpips / ssim / ms_ssim / psnr",
        direction="mixed",
        when_en="Pairwise reconstruction / consistency on aligned image pairs.",
        when_zh="对齐图像对上的成对重建/一致性指标。",
        code="""
import numpy as np
from PIL import Image
from worldfoundry.evaluation.tasks.metrics import (
    compute_lpips,
    compute_ssim,
    compute_ms_ssim,
    compute_psnr,
    compute_perceptual_bundle,
)

ref = np.array(Image.open("ref.png").convert("RGB"))
gen = np.array(Image.open("gen.png").convert("RGB"))

print("lpips", compute_lpips(ref, gen))          # lower better
print("ssim", compute_ssim(ref, gen))            # higher better
print("ms_ssim", compute_ms_ssim(ref, gen))
print("psnr", compute_psnr(ref, gen))
print("bundle", compute_perceptual_bundle(ref, gen))
""",
        returns_en="Float per metric; LPIPS lower, SSIM/MS-SSIM/PSNR higher is better.",
        returns_zh="各 metric 为 float；LPIPS 越低越好，SSIM/MS-SSIM/PSNR 越高越好。",
    ),
    recipe(
        "dino_similarity / dreamsim / fsim / cpbd",
        direction="mixed",
        when_en="Perceptual similarity or no-reference sharpness on numpy images.",
        when_zh="numpy 图像上的感知相似度或无参考锐度。",
        code="""
import numpy as np
from PIL import Image
from worldfoundry.evaluation.tasks.metrics import (
    compute_dino_similarity,
    compute_dreamsim,
    compute_fsim,
    compute_cpbd,
)

ref = np.array(Image.open("ref.png").convert("RGB"))
gen = np.array(Image.open("gen.png").convert("RGB"))

print("dino", compute_dino_similarity(ref, gen))
print("dreamsim", compute_dreamsim(ref, gen))
print("fsim", compute_fsim(ref, gen))
print("cpbd", compute_cpbd(gen))  # single image, no reference
""",
        returns_en="Float scores; DreamSim lower = more similar.",
        returns_zh="浮点分数；DreamSim 越低越相似。",
    ),
    recipe(
        "mask_accuracy",
        direction="↑",
        when_en="Segmentation mask accuracy / IoU between prediction and GT.",
        when_zh="预测 mask 与 GT 的准确率 / IoU。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import compute_mask_accuracy, compute_mask_iou

pred = np.load("pred_mask.npy")  # HxW binary or label map
gt = np.load("gt_mask.npy")
print(compute_mask_accuracy(pred, gt))
print(compute_mask_iou(pred, gt))
""",
        returns_en="Accuracy and IoU floats in `[0, 1]`.",
        returns_zh="`[0, 1]` 的 accuracy 与 IoU。",
    ),
    recipe(
        "object_detection",
        direction="↑",
        when_en="WorldScore-style phrase-matching detection success rate.",
        when_zh="WorldScore 风格短语匹配检测成功率。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_object_detection_success_rate

result = compute_object_detection_success_rate(
    generated_dir="/data/generated",
    prompts_path="/data/prompts.json",
)
print(result)
""",
        returns_en="Success rate dict; see `object_detection/` wrapper for input schema.",
        returns_zh="成功率 dict；输入 schema 见 `object_detection/` wrapper。",
    ),
    recipe(
        "lqs",
        direction="↑",
        when_en="Layout Quality Score over predicted vs GT bounding-box layouts.",
        when_zh="预测 vs GT bbox layout 的 Layout Quality Score。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_lqs

gt_layout = [{"label": "person", "bbox": [10, 20, 110, 220]}]
pred_layout = [{"label": "person", "bbox": [12, 18, 108, 218]}]
print(compute_lqs(gt_layout, pred_layout))
""",
        returns_en="Dict with LR/LP/LC/AC components and combined LQS.",
        returns_zh="含 LR/LP/LC/AC 分量与组合 LQS 的 dict。",
    ),
    recipe(
        "semsr",
        direction="↑",
        when_en="Semantic Shift Rate for trigger/origin/target image triplets.",
        when_zh="trigger/origin/target 三图组的 Semantic Shift Rate。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_semsr_from_images

result = compute_semsr_from_images(
    image_ust="trigger.png",
    image_ori="origin.png",
    image_tar="target.png",
    semantic_text="a photo of a cat",
)
print(result["semsr"])
""",
        returns_en="Dict with `semsr`, `sem_shift`, and CLIP similarities.",
        returns_zh="含 `semsr`、`sem_shift` 与 CLIP 相似度的 dict。",
    ),
    recipe(
        "irs",
        direction="↑",
        when_en="Image Realism Score calibrated against real-image statistics.",
        when_zh="相对真实图统计校准的 Image Realism Score。",
        code="""
from PIL import Image
from worldfoundry.evaluation.tasks.metrics import compute_irs_with_reference

real_paths = ["real/1.jpg", "real/2.jpg"]
test_paths = ["gen/1.jpg", "gen/2.jpg"]
real_images = [Image.open(p) for p in real_paths]
test_images = [Image.open(p) for p in test_paths]

out = compute_irs_with_reference(test_images, real_images)
print(out["irs_mean"], out["irs_scores"])
""",
        returns_en="Dict with per-image IRS and fitted reference means.",
        returns_zh="含逐图 IRS 与拟合 reference means 的 dict。",
    ),
    recipe(
        "cas",
        direction="↑",
        when_en="Train classifier on synthetic images; evaluate Top-k on real labels.",
        when_zh="合成图训练分类器，在真实标签上算 Top-k accuracy。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import (
    train_classifier_and_compute_cas,
    compute_cas_from_predictions,
)

out = train_classifier_and_compute_cas(
    synthetic_images=syn_x,
    synthetic_labels=syn_y,
    real_images=real_x,
    real_labels=real_y,
    num_classes=10,
    device="cuda",
)
print(out["cas_top1"], out.get("cas_top5"))

# or with existing predictions
print(compute_cas_from_predictions(real_y, pred_topk, topk=(1, 5)))
""",
        returns_en="Dict with `cas`, `cas_top1`, optional `cas_top5`.",
        returns_zh="含 `cas`、`cas_top1`、可选 `cas_top5` 的 dict。",
    ),
    recipe(
        "manipulation_direction",
        direction="↑",
        when_en="CLIP change-vector alignment for image editing evaluation.",
        when_zh="图像编辑评估的 CLIP 变化向量对齐。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_manipulation_direction_from_pairs

md = compute_manipulation_direction_from_pairs(
    image_input="before.png",
    image_manipulated="after.png",
    text_original="a red car",
    text_replaced="a blue car",
)
print(md)
""",
        returns_en="Cosine similarity in `[-1, 1]`; higher = edit follows text direction.",
        returns_zh="`[-1, 1]` 余弦相似度；越高说明编辑越沿文本方向。",
    ),
    recipe(
        "vs_similarity",
        direction="↑",
        when_en="HDGAN visual–semantic similarity from paired embeddings.",
        when_zh="成对 embedding 的 HDGAN 视觉–语义相似度。",
        code="""
import numpy as np
from worldfoundry.evaluation.tasks.metrics import compute_vs_similarity

image_embeds = np.load("img_emb.npy")  # (N, D)
text_embeds = np.load("txt_emb.npy")   # (N, D)
print(compute_vs_similarity(image_embeds, text_embeds, paired=True))
""",
        returns_en="Float VS similarity for paired batches.",
        returns_zh="成对 batch 的 VS 相似度 float。",
    ),
    recipe(
        "quality_loss",
        direction="↓",
        when_en="CLIPScore × text-presence probability (prompt optimization metric).",
        when_zh="CLIPScore × 文字出现概率（prompt 优化指标）。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_quality_loss_for_pair

out = compute_quality_loss_for_pair(
    image="sample.png",
    prompt="HELLO WORLD",
    has_text=True,
)
print(out)  # clip_score, text_presence_probability, quality_loss
""",
        returns_en="Dict; lower quality_loss with missing text is expected behavior.",
        returns_zh="dict；无文字时 quality_loss 降低是预期行为。",
    ),
    recipe(
        "object_wise_consistency",
        direction="↑",
        when_en="Location-aware T2I: match guidance boxes to detector outputs.",
        when_zh="位置感知 T2I：guidance box 与检测框匹配。",
        code="""
from worldfoundry.evaluation.tasks.metrics import compute_object_wise_consistency

guidance = [([10, 20, 100, 200], "cat"), ([120, 30, 220, 180], "dog")]
detections = [([12, 22, 98, 198], "cat"), ([125, 35, 215, 175], "dog")]

print(compute_object_wise_consistency(guidance, detections, iou_threshold=0.5))
""",
        returns_en="Dict with mean IoU and success rate `object_wise_success_rate`.",
        returns_zh="含 mean IoU 与 `object_wise_success_rate` 的 dict。",
    ),
]


def render(locale: str) -> str:
    zh = locale == "zh"
    lines: list[str] = []

    if zh:
        lines.extend(
            [
                "## 每个指标怎么用",
                "",
                "三步走：",
                "",
                "1. `source WorldFoundry/tmp/worldfoundry_unified_env.sh` 并 `pip install -e \".[distribution_metrics]\"`",
                "2. `worldfoundry-eval metric show <id>` 确认 id / alias",
                "3. 复制下面对应 metric 的示例代码运行",
                "",
                "只有 **built-in existing-results** 能走 CLI；其余 metric 用 Python `compute_*` 或 Scorer 类。",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## How to use each metric",
                "",
                "Three steps:",
                "",
                "1. `source WorldFoundry/tmp/worldfoundry_unified_env.sh` and `pip install -e \".[distribution_metrics]\"`",
                "2. `worldfoundry-eval metric show <id>` to resolve ids and aliases",
                "3. Copy the recipe below for your metric and run it",
                "",
                "Only **built-in existing-results** metrics use `worldfoundry-eval evaluate`; everything else is Python (`compute_*` or scorer classes).",
                "",
            ]
        )

    for item in RECIPES:
        mid = item["id"]
        direction = item["direction"]
        when = item["when_zh"] if zh else item["when_en"]
        returns = item["returns_zh"] if zh else item["returns_en"]
        extra = item["extra_zh"] if zh else item["extra_en"]
        label = "用法" if zh else "When"
        ret_label = "返回" if zh else "Returns"

        lines.append(f"### `{mid}` ({direction})")
        lines.append("")
        lines.append(f"**{label}**: {when}")
        lines.append("")
        lines.append("```bash" if item["code"].lstrip().startswith("worldfoundry-eval") else "```python")
        lines.append(item["code"])
        lines.append("```")
        lines.append("")
        lines.append(f"**{ret_label}**: {returns}")
        if extra:
            lines.append("")
            lines.append(extra)
        lines.append("")

    if zh:
        lines.extend(
            [
                "### 一次跑多个 torch-fidelity 指标",
                "",
                "```python",
                "from worldfoundry.evaluation.tasks.metrics import compute_distribution_metrics",
                "",
                "scores = compute_distribution_metrics(",
                '    "/data/ref", "/data/gen", metrics=("fid", "kid", "prc")',
                ")",
                "print(scores)",
                "```",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "### Run multiple torch-fidelity metrics at once",
                "",
                "```python",
                "from worldfoundry.evaluation.tasks.metrics import compute_distribution_metrics",
                "",
                "scores = compute_distribution_metrics(",
                '    "/data/ref", "/data/gen", metrics=("fid", "kid", "prc")',
                ")",
                "print(scores)",
                "```",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    OUT_EN.write_text(render("en"), encoding="utf-8")
    OUT_ZH.write_text(render("zh"), encoding="utf-8")
    print(f"wrote {OUT_EN}")
    print(f"wrote {OUT_ZH}")


if __name__ == "__main__":
    main()
