import json
import sys
from typing import Any, Dict

def summarize_value(v: Any) -> Any:
    if isinstance(v, list):
        return {"type": "list", "length": len(v)}
    if isinstance(v, dict):
        return {"type": "object", "keys": list(v.keys())}
    return {"type": type(v).__name__}

def extract_metadata(data: Any) -> Dict[str, Any]:
    if isinstance(data, list):
        return {
            "root_type": "list",
            "num_items": len(data),
            "item_structure": extract_metadata(data[0]) if data else {}
        }

    if isinstance(data, dict):
        meta = {"root_type": "object", "fields": {}}
        for k, v in data.items():
            meta["fields"][k] = summarize_value(v)
        return meta

    return {"root_type": type(data).__name__}

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_metadata.py <file.json>")
        return

    path = sys.argv[1]
    with open(path, "r") as f:
        data = json.load(f)

    metadata = extract_metadata(data)
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()

