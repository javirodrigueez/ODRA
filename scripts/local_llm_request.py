import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


def load_request(path: Path, model: str) -> dict:
    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "model": model,
            "messages": [{"role": "user", "content": text}],
        }

    if isinstance(data, dict):
        return {"model": model, **data}
    if isinstance(data, list):
        return {"model": model, "messages": data}

    raise ValueError("Request file must be plain text, a JSON object, or a JSON messages list.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a file-backed request to a local OpenAI-compatible LLM."
    )
    parser.add_argument("request_file", type=Path, help="Prompt text or JSON request file.")
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", "local-model"))
    parser.add_argument("--base-url", default=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("LOCAL_LLM_API_KEY", "local-no-key-required"))
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    request = load_request(args.request_file, args.model)

    response = client.chat.completions.create(**request)
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
