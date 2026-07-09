"""Module for base_models -> perception_core -> detection -> grounding_dino -> util -> get_tokenlizer.py functionality."""

import os

from transformers import AutoTokenizer, BertModel, RobertaModel


def get_tokenlizer(text_encoder_type):
    """Get tokenlizer.

    Args:
        text_encoder_type: The text encoder type.
    """
    if not isinstance(text_encoder_type, str):
        # print("text_encoder_type is not a str")
        if hasattr(text_encoder_type, "text_encoder_type"):
            text_encoder_type = text_encoder_type.text_encoder_type
        elif text_encoder_type.get("text_encoder_type", False):
            text_encoder_type = text_encoder_type.get("text_encoder_type")
        elif os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type):
            pass
        else:
            raise ValueError(
                "Unknown type of text_encoder_type: {}".format(type(text_encoder_type))
            )
    return AutoTokenizer.from_pretrained(text_encoder_type)


def get_pretrained_language_model(text_encoder_type):
    """Get pretrained language model.

    Args:
        text_encoder_type: The text encoder type.
    """
    if text_encoder_type == "bert-base-uncased" or (os.path.isdir(text_encoder_type) and os.path.exists(text_encoder_type)):
        return BertModel.from_pretrained(text_encoder_type)
    if text_encoder_type == "roberta-base":
        return RobertaModel.from_pretrained(text_encoder_type)

    raise ValueError("Unknown text_encoder_type {}".format(text_encoder_type))
