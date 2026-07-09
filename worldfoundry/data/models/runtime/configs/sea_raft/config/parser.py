"""Parse SEA-RAFT configuration files into ``argparse.Namespace`` objects.

Utilities for loading JSON configuration files and merging them with
command-line arguments parsed via ``argparse``.  CLI overrides take
precedence over values read from the JSON file.
"""

import json
import argparse


def json_to_args(json_path):
    """Load a JSON configuration file into an ``argparse.Namespace`` object.

    Args:
        json_path: Path to the JSON configuration file.

    Returns:
        :class:`argparse.Namespace`: Namespace whose attributes correspond
        to the top-level keys in the JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    args = argparse.Namespace()
    args_dict = args.__dict__
    for key, value in data.items():
        args_dict[key] = value
    return args


def parse_args(parser):
    """Parse CLI arguments and merge them with a JSON config file.

    The ``cfg`` CLI argument specifies the JSON file path.  Any other
    CLI flags override the corresponding entries loaded from that file.

    Args:
        parser: :class:`argparse.ArgumentParser` with a ``cfg`` argument
            and any additional model-specific flags.

    Returns:
        :class:`argparse.Namespace`: Merged configuration namespace where
        CLI values take precedence over JSON values.
    """
    entry = parser.parse_args()
    json_path = entry.cfg
    args = json_to_args(json_path)
    args_dict = args.__dict__
    # NOTE: CLI arguments override JSON config entries.
    for index, (key, value) in enumerate(vars(entry).items()):
        args_dict[key] = value
    return args
