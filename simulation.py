from graph.graph import TherapyGraphBuilder
import argparse
from copy import deepcopy
import json
import os
from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
from typing import cast

from graph.state import ConversationMessage
from runtime_config import load_runtime_config


def _load_profiles(profiles_path: str) -> list[dict]:
  if profiles_path.endswith(".jsonl"):
    profiles = []
    with open(profiles_path, "r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        profiles.append(json.loads(line))
    return profiles

  with open(profiles_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  if isinstance(payload, list):
    return payload
  if isinstance(payload, dict):
    return [payload]
  raise ValueError(f"Unsupported profiles payload in {profiles_path}")


def _load_resume_state(resume_path: str) -> dict:
  with open(resume_path, "r", encoding="utf-8") as f:
    payload = json.load(f)
  if not isinstance(payload, dict):
    raise ValueError(f"Unsupported resume payload in {resume_path}: expected a JSON object.")
  profile = payload.get("patient_profile")
  if not isinstance(profile, dict):
    raise ValueError(f"Resume file {resume_path} is missing a valid patient_profile object.")
  return payload


def run_simulation(
  cfg_path: str,
  profile: dict,
  max_turns: int,
  min_turns: int,
  output_dir: str = '',
  verbose: bool = False,
  resume_state: dict | None = None,
):
  """Run a single simulation for a given profile."""
  result = {
    "messages": [],
    "exchange_counts": {},
    "utterance_index": 0,
    "session_ended": True,
    "session_ended_reason": "",
    "conversation_error": "",
  }
  error_printed = False
  try:
    builder = TherapyGraphBuilder(cfg_path, profile)
    chain = builder.create_chain()
    if resume_state is None:
      initial_state = builder.initial_state(max_turns=max_turns, min_turns=min_turns)
    else:
      initial_state = builder.resume_state(deepcopy(resume_state), max_turns=max_turns, min_turns=min_turns)
    result = initial_state

    if verbose:
      print()
      last_msg_count = len(initial_state.get("messages", []))
      for chunk in chain.stream(initial_state, stream_mode="values"):
        messages = chunk.get("messages", [])
        current_stage = chunk.get("current_stage", "N/A")
        r = chunk.get("resistance", "N/A")
        pred_r = chunk.get("predicted_resistance", "N/A")
        for msg in messages[last_msg_count:]:
          # if msg["role"] == "therapist":
          #   #print(f"<MI-TECH-LOG>{chunk.get("mi_techniques_used")[-2:]}</MI-TECH-LOG>")
          #   print(f"<STACK>{chunk.get("mi_topic_stack")}</STACK>")
          # elif msg["role"] == "patient":
          #   print(f"<ACTION>{chunk.get("last_patient_action_distribution")}</ACTION>")
          r_print = pred_r if msg["role"] == "therapist" else r
          print(f'    {msg["role"].capitalize()} ({r_print}) [{current_stage}]: {msg["content"]}')
        last_msg_count = len(messages)
        result = chunk
      if result.get("conversation_error"):
        print(f'ERROR: {result["conversation_error"]}')
        error_printed = True
      print()
    else:
      result = chain.invoke(initial_state)
  except Exception as exc:
    result = dict(result)
    result["session_ended"] = True
    result["session_ended_reason"] = result.get("session_ended_reason") or "conversation_error"
    result["conversation_error"] = str(exc)
    if verbose:
      print(f'ERROR: {result["conversation_error"]}')
      error_printed = True

  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    sim_idx = profile.get("idx", "unknown")
    transcript_file = os.path.join(output_dir, f"simulation_{sim_idx}.txt")
    json_file = os.path.join(output_dir, f"simulation_{sim_idx}.json")
    with open(transcript_file, "w", encoding="utf-8") as f:
      messages = cast(list[ConversationMessage], result.get("messages", []))
      for message in messages:
        f.write(f'--> {message["role"].capitalize()}: {message["content"]}\n\n')
    with open(json_file, "w", encoding="utf-8") as f:
      json.dump(result, f, indent=2, ensure_ascii=True)
    print(f"  Saved transcript: {transcript_file}")
    print(f"  Saved state JSON: {json_file}")

  if verbose and result.get("conversation_error") and not error_printed:
    print(f'ERROR: {result["conversation_error"]}')

  return result


def main(cfg_path: str):
  load_dotenv()

  cfg = load_runtime_config(cfg_path)

  max_turns = cfg.env.max_turns
  min_turns = cfg.env.get("min_turns", 0)
  output_dir = cfg.env.output_dir
  verbose = cfg.env.verbose
  resume_file = cfg.env.get("resume_file")

  if resume_file:
    resume_state = _load_resume_state(resume_file)
    profile = deepcopy(resume_state["patient_profile"])
    profile.setdefault("idx", "unknown")
    print(f"Resuming simulation from {resume_file}\n")
    result = run_simulation(
      cfg_path,
      profile,
      max_turns,
      min_turns,
      output_dir,
      verbose,
      resume_state=resume_state,
    )
    exchange_counts = cast(dict[str, list[int]], result.get("exchange_counts", {}))
    exchanges = sum(item for sublist in exchange_counts.values() for item in sublist)
    print(f'  Total utterances: {result.get("utterance_index", 0)}')
    print(f"  Total exchanges: {exchanges}")
    print(f'  End reason: {result.get("session_ended_reason", "unknown")}\n')
    return

  profiles_path = cfg.patient.profiles
  profiles = _load_profiles(profiles_path)

  print(f"Loaded {len(profiles)} profiles from {profiles_path}\n")

  for i, profile in enumerate(profiles):
    if "idx" not in profile:
      profile["idx"] = i + 1
    print(f'{"="*60}')
    print(f'Simulation {i+1}/{len(profiles)} — idx: {profile["idx"]}')
    print(f'{"="*60}')

    result = run_simulation(cfg_path, profile, max_turns, min_turns, output_dir, verbose)
    exchange_counts = cast(dict[str, list[int]], result.get("exchange_counts", {}))
    exchanges = sum(item for sublist in exchange_counts.values() for item in sublist)
    print(f'  Total utterances: {result.get("utterance_index", 0)}')
    print(f"  Total exchanges: {exchanges}")
    print(f'  End reason: {result.get("session_ended_reason", "unknown")}\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run CBT therapy simulation")
  parser.add_argument("--cfg", type=str, default="configs/bedrock.yaml", help="YAML configuration file.")
  args = parser.parse_args()
  main(args.cfg)
