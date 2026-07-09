"""Module for base_models -> perception_core -> detection -> grounding_dino -> util -> time_counter.py functionality."""

import json
import time


class TimeCounter:
    """Time counter implementation."""
    def __init__(self) -> None:
        """Init.

        Returns:
            The return value.
        """
        pass

    def clear(self):
        """Clear."""
        self.timedict = {}
        self.basetime = time.perf_counter()

    def timeit(self, name):
        """Timeit.

        Args:
            name: The name.
        """
        nowtime = time.perf_counter() - self.basetime
        self.timedict[name] = nowtime
        self.basetime = time.perf_counter()


class TimeHolder:
    """Time holder implementation."""
    def __init__(self) -> None:
        """Init.

        Returns:
            The return value.
        """
        self.timedict = {}

    def update(self, _timedict: dict):
        """Update.

        Args:
            _timedict: The timedict.
        """
        for k, v in _timedict.items():
            if k not in self.timedict:
                self.timedict[k] = AverageMeter(name=k, val_only=True)
            self.timedict[k].update(val=v)

    def final_res(self):
        """Final res."""
        return {k: v.avg for k, v in self.timedict.items()}

    def __str__(self):
        """Str."""
        return json.dumps(self.final_res(), indent=2)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", val_only=False):
        """Init.

        Args:
            name: The name.
            fmt: The fmt.
            val_only: The val only.
        """
        self.name = name
        self.fmt = fmt
        self.val_only = val_only
        self.reset()

    def reset(self):
        """Reset."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update.

        Args:
            val: The val.
            n: The n.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """Str."""
        if self.val_only:
            fmtstr = "{name} {val" + self.fmt + "}"
        else:
            fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
