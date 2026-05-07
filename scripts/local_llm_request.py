import argparse
import json
import os
from pathlib import Path

from openai import OpenAI


DEFAULT_MAX_MODEL_LEN = 131072


def load_request(path: Path, model: str, max_model_len: int) -> dict:
    open = '''[opening_stage_initial_greeting_reasoning]
```json
{
  "reasoning": "The session is just beginning. The patient's intake form indicates a history of mild anxiety and a request for online support. The initial goal is to establish a safe, 
empathetic connection. A simple, open-ended greeting is appropriate to encourage the patient to share their current state."
}
```
</think>

Hello, I'm glad to meet you. How are you feeling today?
'''
    text = path.read_text(encoding="utf-8")
    system = Path("./agent_prompts/therapist_system_ft.txt").read_text(encoding="utf-8")

    return {
      "model": model,
      "max_tokens": max_model_len,
      "messages": [
          {"role": "system", "content": system},
          {"role": "assistant", "content": open},
          {"role": "user", "content": "Hi, well. I have been feeling a bit anxious lately, especially with work and family responsibilities piling up."},
      ],
    }

def print_streaming_response(response) -> None:
    for chunk in response:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a file-backed request to a local OpenAI-compatible LLM."
    )
    parser.add_argument("request_file", type=Path, help="Prompt text or JSON request file.")
    parser.add_argument("--model", default=os.getenv("LOCAL_LLM_MODEL", "ft_models/Qwen3.6-27B-ctxt_rtraces/ft_merged/"))
    parser.add_argument("--base-url", default=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("LOCAL_LLM_API_KEY", "local-no-key-required"))
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=int(os.getenv("LOCAL_LLM_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN)),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Print generated tokens as they arrive.",
    )
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    request = load_request(args.request_file, args.model, args.max_model_len)
    if args.stream:
        request["stream"] = True

    response = client.chat.completions.create(**request)
    if args.stream:
        print_streaming_response(response)
        return

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
