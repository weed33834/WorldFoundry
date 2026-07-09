from abc import ABC


class BaseRepresentation(ABC):
    """Base contract for representation backends."""

    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, pretrained_model_path, device=None, **kwargs):
        raise NotImplementedError(f"{cls.__name__}.from_pretrained() must be implemented by subclasses.")

    def api_init(self, api_key, endpoint):
        raise NotImplementedError(f"{type(self).__name__}.api_init() must be implemented by subclasses.")

    def get_representation(self, data):
        raise NotImplementedError(f"{type(self).__name__}.get_representation() must be implemented by subclasses.")
