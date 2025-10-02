from __future__ import annotations
"""Config loader with env expansion.

Reads YAML, expands ${VAR} from environment if present.
"""
from typing import Dict, Any
import os
import re
import yaml
from dotenv import load_dotenv

load_dotenv()

env_var_pat = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.getenv(var, "")
        return env_var_pat.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(x) for x in value]
    return value


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _expand_env(cfg)
