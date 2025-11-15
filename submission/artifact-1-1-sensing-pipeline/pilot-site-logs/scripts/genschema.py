import json
import sys
from typing import Any, Dict, List, Union

def merge_types(a: Union[str, List[str]], b: str) -> List[str]:
    if isinstance(a, list):
        if b not in a:
            a.append(b)
        return a
    if a == b:
        return a
    return list({a, b})

def infer_schema(value: Any) -> Dict[str, Any]:
    if value is None:
        return {"type": "null"}

    t = type(value)

    if t in (int, float, str, bool):
        return {"type": t.__name__}

    if t is list:
        if not value:
            return {"type": "array", "items": {}}

        # infer schema for each element and merge
        item_schemas = [infer_schema(v) for v in value]
        merged = item_schemas[0]

        for s in item_schemas[1:]:
            for k in s:
                if k == "type":
                    merged["type"] = merge_types(merged["type"], s["type"])
                else:
                    merged[k] = s[k]

        return {"type": "array", "items": merged}

    if t is dict:
        return {
            "type": "object",
            "properties": {k: infer_schema(v) for k, v in value.items()}
        }

    return {"type": "string"}  # fallback

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_schema.py <file.json>")
        return

    path = sys.argv[1]
    with open(path, "r") as f:
        data = json.load(f)

    schema = infer_schema(data)
    print(json.dumps(schema, indent=2))

if __name__ == "__main__":
    main()

