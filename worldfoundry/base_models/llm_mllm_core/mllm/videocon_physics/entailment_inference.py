from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.models.llama.tokenization_llama import LlamaTokenizer

from worldfoundry.base_models.llm_mllm_core.mllm.videocon_physics.data import (
    MultiModalDataset,
    batchify,
)
from worldfoundry.base_models.llm_mllm_core.mllm.videophy2_autoeval.mplug_owl_video.modeling_mplug_owl import (
    MplugOwlForConditionalGeneration,
)
from worldfoundry.base_models.llm_mllm_core.mllm.videophy2_autoeval.mplug_owl_video.processing_mplug_owl import (
    MplugOwlImageProcessor,
    MplugOwlProcessor,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run in-tree VideoCon-Physics entailment inference.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def _get_entailment_scores(logits: torch.Tensor, input_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    softmax = nn.Softmax(dim=2)
    logits = softmax(logits)
    token_id_yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
    token_id_no = tokenizer.encode("No", add_special_tokens=False)[0]
    scores: list[torch.Tensor] = []
    for row_index in range(len(logits)):
        answer_index = len(input_ids[row_index]) - 1
        for token_index in range(len(input_ids[row_index])):
            if input_ids[row_index][token_index] == tokenizer.pad_token_id:
                answer_index = token_index - 1
                break
        yes_prob = logits[row_index][answer_index][token_id_yes]
        no_prob = logits[row_index][answer_index][token_id_no]
        scores.append(yes_prob / (yes_prob + no_prob))
    return torch.stack(scores)


def _move_batch_to_device(inputs: dict[str, object], device: torch.device) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            tensor = value.bfloat16() if value.dtype == torch.float else value
            moved[key] = tensor.to(device)
        else:
            moved[key] = value
    return moved


def _score_rows(model, tokenizer, dataloader: DataLoader) -> list[list[object]]:
    rows: list[list[object]] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = _move_batch_to_device(batch, device)
            outputs = model(
                pixel_values=inputs["pixel_values"],
                video_pixel_values=inputs["video_pixel_values"],
                labels=None,
                num_images=inputs["num_images"],
                num_videos=inputs["num_videos"],
                input_ids=inputs["input_ids"],
                non_padding_mask=inputs["non_padding_mask"],
                non_media_mask=inputs["non_media_mask"],
                prompt_mask=inputs["prompt_mask"],
            )
            entail_scores = _get_entailment_scores(outputs["logits"], inputs["input_ids"], tokenizer)
            for index, score in enumerate(entail_scores):
                rows.append([inputs["videopaths"][index], inputs["captions"][index], float(score.item())])
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = LlamaTokenizer.from_pretrained(str(checkpoint))
    image_processor = MplugOwlImageProcessor.from_pretrained(str(checkpoint))
    processor = MplugOwlProcessor(image_processor, tokenizer)
    dataset = MultiModalDataset(
        str(args.input_csv.expanduser().resolve()),
        tokenizer,
        processor,
        max_length=args.max_length,
        num_frames=args.num_frames,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, collate_fn=batchify)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        str(checkpoint),
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    model.eval()

    rows = _score_rows(model, tokenizer, dataloader)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
