"""
Optional machine-wide settings, e.g. for a shared server.

Read from the TOML file named by $SSCD_SETTINGS, else ~/.sscd/settings.toml.
Without a settings file the defaults below apply (single-user laptop use).

Example settings.toml:

    # where all runs and evaluations are saved
    output_root = "/data/sscd_outputs"
    # the GUI only accepts scale folders inside these folders (empty: any folder)
    allowed_input_roots = ["/mnt/scale_archive"]
    # how many runs may execute at the same time; further runs wait in a queue
    max_concurrent_runs = 1
"""

import os
import tomllib
from pathlib import Path

DEFAULTS = {
    "output_root": None,          # None: $SSCD_OUTPUT_ROOT, else ~/sscd_outputs
    "allowed_input_roots": [],    # empty: any folder
    "max_concurrent_runs": 1,
}


def settings_path():
    return Path(os.environ.get("SSCD_SETTINGS") or Path.home() / ".sscd" / "settings.toml").expanduser()


def load_settings():
    """The settings, with defaults for anything not set. Raises on an invalid file."""
    settings = dict(DEFAULTS)
    path = settings_path()
    if path.is_file():
        with open(path, "rb") as f:
            user = tomllib.load(f)
        unknown = set(user) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown setting(s) in {path}: {', '.join(sorted(unknown))}")
        settings.update(user)
    settings["allowed_input_roots"] = [str(Path(p).expanduser().resolve())
                                       for p in settings["allowed_input_roots"]]
    settings["max_concurrent_runs"] = max(1, int(settings["max_concurrent_runs"]))
    return settings


def input_dir_allowed(path, settings):
    """True if `path` is inside one of the allowed input roots (or none are configured)."""
    roots = settings["allowed_input_roots"]
    if not roots:
        return True
    path = Path(path).expanduser().resolve()
    return any(path == Path(r) or Path(r) in path.parents for r in roots)
