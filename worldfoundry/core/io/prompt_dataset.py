"""Small line-oriented prompt datasets shared by inference runtimes."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset


class TextPromptDataset(Dataset):
    """Read one prompt per line with an optional aligned extended-prompt file."""

    def __init__(self, prompt_path: str | Path, extended_prompt_path: str | Path | None = None) -> None:
        self.prompt_list = Path(prompt_path).read_text(encoding="utf-8").splitlines()
        if extended_prompt_path is None:
            self.extended_prompt_list = None
        else:
            self.extended_prompt_list = Path(extended_prompt_path).read_text(encoding="utf-8").splitlines()
            if len(self.extended_prompt_list) != len(self.prompt_list):
                raise ValueError(
                    "Prompt and extended-prompt files must contain the same number of lines: "
                    f"{len(self.prompt_list)} != {len(self.extended_prompt_list)}"
                )

    def __len__(self) -> int:
        return len(self.prompt_list)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample: dict[str, object] = {"prompts": self.prompt_list[index], "idx": index}
        if self.extended_prompt_list is not None:
            sample["extended_prompts"] = self.extended_prompt_list[index]
        return sample


__all__ = ["TextPromptDataset"]
