import json
import os
import shlex
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any


def _to_serializable(value: Any):
    if is_dataclass(value):
        return {k: _to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def save_training_command(output_dir: str, script_name: str, **sections: Any) -> str:
    os.makedirs(output_dir, exist_ok=True)

    command = " ".join(shlex.quote(arg) for arg in sys.argv)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cwd": os.getcwd(),
        "script": script_name,
        "output_dir": output_dir,
        "command": command,
        "sections": {name: _to_serializable(value) for name, value in sections.items()},
    }

    text_lines = [
        f"timestamp: {payload['timestamp']}",
        f"cwd: {payload['cwd']}",
        f"script: {payload['script']}",
        f"output_dir: {payload['output_dir']}",
        "",
        "command:",
        payload["command"],
        "",
        "parsed_args:",
        json.dumps(payload["sections"], ensure_ascii=False, indent=2),
        "",
    ]

    text_path = os.path.join(output_dir, "run_command.txt")
    json_path = os.path.join(output_dir, "run_command.json")

    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return text_path
