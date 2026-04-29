from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


CORE_BELIEF_KEYS = ["helplessness", "unlovability", "worthlessness"]
DISTORTION_KEYS = [
  "mental_filtering",
  "fortune_telling",
  "mind_reading",
  "personalization",
  "catastrophizing",
  "all_or_nothing_thinking",
  "should_statements",
  "overgeneralization",
  "labeling_and_mislabeling",
  "discounting_the_positive",
]


class DatasetBuildError(ValueError):
  pass


def ensure_str(value: Any) -> str:
  if value is None:
    return ""
  return str(value)


def require_dict(value: Any, label: str) -> dict[str, Any]:
  if not isinstance(value, dict):
    raise DatasetBuildError(f"{label} must be a JSON object.")
  return value


def require_list(value: Any, label: str) -> list[Any]:
  if not isinstance(value, list):
    raise DatasetBuildError(f"{label} must be a JSON list.")
  return value


def compact_json(value: Any) -> str:
  return json.dumps(value, ensure_ascii=False, sort_keys=True)


def format_patient_intake_form(profile: dict[str, Any]) -> str:
  personal_info = profile.get("personal_info", {})
  more_info = profile.get("more_information", {})

  sections: list[str] = []
  if isinstance(personal_info, dict) and personal_info:
    personal_lines = [
      f"{str(key).replace('_', ' ').title()}: {value}"
      for key, value in personal_info.items()
    ]
    sections.append("Personal Info:\n" + "\n".join(personal_lines))

  reason = profile.get("reason_seeking_counseling", "")
  if reason:
    sections.append(f"Reason Seeking Counseling:\n{reason}")

  if isinstance(more_info, dict) and more_info:
    more_info_lines = [
      f"{str(key).replace('_', ' ').title()}:\n{value}"
      for key, value in more_info.items()
      if value
    ]
    if more_info_lines:
      sections.append("More Information:\n" + "\n\n".join(more_info_lines))

  return "\n\n".join(sections)


def project_root() -> Path:
  return Path(__file__).resolve().parents[2]


def load_therapist_system_prompt(root: Path) -> str:
  prompt_path = root / "prompts" / "therapist_system.txt"
  if not prompt_path.is_file():
    raise DatasetBuildError(f"Missing therapist system prompt: {prompt_path}")
  return prompt_path.read_text(encoding="utf-8")


def render_instruction(system_prompt: str, patient_profile: dict[str, Any]) -> str:
  intake = format_patient_intake_form(patient_profile)
  return system_prompt.replace("{patient_intake_form}", intake)


def parse_trace_output(trace: dict[str, Any]) -> Any:
  output = trace.get("output")
  if isinstance(output, str):
    stripped = output.strip()
    if not stripped:
      return ""
    try:
      return json.loads(stripped)
    except json.JSONDecodeError:
      return stripped
  return output


def value_text(value: Any, *, empty: str = "none") -> str:
  if value is None:
    return empty
  if isinstance(value, str):
    stripped = value.strip()
    return stripped if stripped else empty
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, (int, float)):
    return str(value)
  if isinstance(value, list):
    if not value:
      return empty
    return "; ".join(value_text(item, empty=empty) for item in value)
  if isinstance(value, dict):
    if not value:
      return empty
    return compact_json(value)
  return str(value)


def format_probability_pairs(payload: dict[str, Any], keys: list[str] | None = None) -> str:
  selected_keys = keys or list(payload.keys())
  pairs: list[str] = []
  for key in selected_keys:
    if key in payload:
      pairs.append(f"{key}={value_text(payload.get(key), empty='null')}")
  if not pairs:
    pairs = [
      f"{key}={value_text(value, empty='null')}"
      for key, value in payload.items()
      if key != "reasoning"
    ]
  return ", ".join(pairs) if pairs else "none"


def highest_probability_key(payload: dict[str, Any], keys: list[str] | None = None) -> str:
  selected_keys = keys or list(payload.keys())
  scored: list[tuple[str, float]] = []
  for key in selected_keys:
    if key not in payload:
      continue
    try:
      scored.append((key, float(payload[key])))
    except (TypeError, ValueError):
      continue
  if not scored:
    return "unknown"
  return max(scored, key=lambda item: item[1])[0]


def fallback_reasoning(step: str, payload: Any, trace: dict[str, Any]) -> str:
  input_text = value_text(trace.get("input"), empty="none")

  if step == "opening_info_update" and isinstance(payload, dict):
    return (
      "I updated the opening context from the patient response: "
      f"mood={value_text(payload.get('mood'), empty='null')}; "
      f"problem={value_text(payload.get('problem'), empty='null')}."
    )

  if step in {"cognitive_gap_planning", "cognitive_refinement_planning"} and isinstance(payload, dict):
    target = ensure_str(payload.get("target")).strip()
    if target:
      return (
        f"I selected {target} as the cognitive refinement target and planned: "
        f"{value_text(payload.get('plan'))}."
      )
    if "new problematic situation" in input_text.lower() or "discover a new" in input_text.lower():
      return f"I planned how to discover a new concrete problem situation: {value_text(payload.get('plan'))}."
    return f"I planned how to fill the missing cognitive model items: {value_text(payload.get('plan'))}."

  if step == "core_belief_probability" and isinstance(payload, dict):
    return (
      "I assigned core belief probabilities: "
      f"{format_probability_pairs(payload, CORE_BELIEF_KEYS)}. "
      f"The highest-probability core belief is {highest_probability_key(payload, CORE_BELIEF_KEYS)}."
    )

  if step == "distortion_probability" and isinstance(payload, dict):
    return (
      "I assigned cognitive distortion probabilities: "
      f"{format_probability_pairs(payload, DISTORTION_KEYS)}. "
      f"The highest-probability distortion is {highest_probability_key(payload, DISTORTION_KEYS)}."
    )

  if step == "address_tib_plan_generation" and isinstance(payload, dict):
    return f"I created a plan for addressing the active therapy-interfering behavior: {value_text(payload.get('tib_plan'))}."

  if step == "additional_technique_selection" and isinstance(payload, dict):
    return (
      f"I selected an additional CBT technique: {value_text(payload.get('name'))}. "
      f"Target: {value_text(payload.get('target_distortion'))}. "
      f"Rationale: {value_text(payload.get('rationale'))}."
    )

  if step == "technique_selection" and isinstance(payload, dict):
    selected = payload.get("selected_techniques")
    if isinstance(selected, list) and selected:
      rendered = []
      for item in selected:
        if isinstance(item, dict):
          rendered.append(
            f"{value_text(item.get('name'))} targeting {value_text(item.get('target_distortion'))} "
            f"because {value_text(item.get('rationale'))}"
          )
        else:
          rendered.append(value_text(item))
      return f"I selected CBT techniques for the session: {'; '.join(rendered)}."
    return "I selected CBT techniques for the session: none."

  if step == "technique_plan" and isinstance(payload, dict):
    plan = payload.get("plan_description") or payload.get("detailed_plan_description")
    return (
      f"I planned the current CBT technique: {value_text(plan)}. "
      f"Expected outcome: {value_text(payload.get('expected_outcome'))}. "
      f"Current progress: {value_text(payload.get('current_progress'))}."
    )

  if step == "technique_evaluation":
    if isinstance(payload, dict) and "evaluation" in payload:
      return f"I evaluated the completed CBT technique: {value_text(payload.get('evaluation'))}."
    return f"I evaluated the completed CBT technique: {value_text(payload)}."

  if step == "closing_feedback_check" and isinstance(payload, dict):
    return (
      "I checked the patient's feedback: "
      f"improvement_requested={value_text(payload.get('improvement_requested'), empty='false')}; "
      f"improvement_description={value_text(payload.get('improvement_description'))}; "
      f"improvement_plan={value_text(payload.get('improvement_plan'))}."
    )

  if step == "closing_summary_generation" and isinstance(payload, dict):
    return f"I generated the closing summary to share with the patient: {value_text(payload.get('session_summary'))}."

  if step == "next_session_plan_generation" and isinstance(payload, dict):
    return f"I generated the next-session plan: {value_text(payload.get('next_session_plan'))}."

  if step == "mi_evaluation" and isinstance(payload, dict):
    return (
      "I evaluated the MI intervention: "
      f"reason_for_mi={value_text(payload.get('reason_for_mi'))}; "
      f"conclusions={value_text(payload.get('conclusions'))}; "
      f"patient_state_change={value_text(payload.get('patient_state_change'))}."
    )

  if step == "address_tib_exit" and isinstance(payload, dict):
    return (
      "I completed the TIB detour: "
      f"tib_name={value_text(payload.get('tib_name'))}; "
      f"exit_reason={value_text(payload.get('exit_reason'))}; "
      f"completion_note={value_text(payload.get('completion_note'))}; "
      f"predicted_resistance={value_text(payload.get('predicted_resistance'), empty='null')}; "
      f"band={value_text(payload.get('band'), empty='null')}; "
      f"resumed from {value_text(payload.get('resume_stage'))}/{value_text(payload.get('resume_substage'))} "
      f"and routed to {value_text(payload.get('routed_stage'))}/{value_text(payload.get('routed_substage'))}; "
      f"resume_context_active={value_text(payload.get('resume_context_active'), empty='false')}; "
      f"resume_context_subject={value_text(payload.get('resume_context_subject'))}."
    )

  if step == "address_tib_repeat_inline_suppressed" and isinstance(payload, dict):
    return (
      "I suppressed repeated TIB rerouting and kept it inline: "
      f"tib_name={value_text(payload.get('tib_name'))}; "
      f"current_stage={value_text(payload.get('current_stage'))}; "
      f"current_substage={value_text(payload.get('current_substage'))}; "
      f"remembered_tib_name={value_text(payload.get('remembered_tib_name'))}; "
      "remembered_tib_no_reroute_active="
      f"{value_text(payload.get('remembered_tib_no_reroute_active'), empty='false')}."
    )

  if step == "technique_auto_complete":
    return f"I marked the current CBT technique complete automatically. Context: {input_text}. Result: {value_text(payload)}."

  if step == "meta_conversation_end_session_redirect":
    return f"I redirected the patient's end-session signal into meta-conversation. Patient signal: {input_text}. Result: {value_text(payload)}."

  if step == "meta_conversation_turn_cap":
    return f"I closed the meta-conversation because the turn cap was reached. Context: {input_text}. Result: {value_text(payload)}."

  if step == "meta_conversation_end_session_repeat":
    return f"I routed a repeated end-session request during meta-conversation to closing fallback. Patient signal: {input_text}. Result: {value_text(payload)}."

  if step == "tib_avoidance_new_cognitive_set":
    return f"I handled avoidance by changing the active cognitive topic. Context: {input_text}. Result: {value_text(payload)}."

  if step == "tib_avoidance_new_technique":
    return f"I handled avoidance by changing the active CBT technique. Context: {input_text}. Result: {value_text(payload)}."

  if step == "anticipated_closing":
    return f"I moved to anticipated closing. Context: {input_text}. Result: {value_text(payload)}."

  if isinstance(payload, (dict, list)):
    return f"I produced this structured output: {compact_json(payload)}."
  return f"I produced this result: {value_text(payload)}."


def trace_reasoning_text(trace: dict[str, Any]) -> str:
  step = ensure_str(trace.get("step")).strip() or "unknown"
  payload = parse_trace_output(trace)
  if isinstance(payload, dict):
    reasoning = ensure_str(payload.get("reasoning")).strip()
    if reasoning:
      return f"[{step}] {reasoning}"
  return f"[{step}] {fallback_reasoning(step, payload, trace)}"


def format_scalar_list(values: Any) -> str:
  if not isinstance(values, list) or not values:
    return "none"
  rendered = [value_text(value) for value in values if value_text(value)]
  return "; ".join(rendered) if rendered else "none"


def format_probability_map(value: Any) -> str:
  if not isinstance(value, dict) or not value:
    return "none"
  return ", ".join(f"{key}={value_text(val, empty='null')}" for key, val in value.items())


def format_cognitive_set(item: Any) -> str:
  if not isinstance(item, dict):
    return value_text(item)
  set_id = value_text(item.get("cognitive_model_set_id"), empty="unknown")
  return (
    f"Cognitive model set {set_id}. "
    f"Situation: {value_text(item.get('situation'))}. "
    f"Automatic thoughts: {format_scalar_list(item.get('automatic_thoughts'))}. "
    f"Emotions: {format_scalar_list(item.get('emotions'))}. "
    f"Behaviors: {format_scalar_list(item.get('behaviors'))}. "
    f"Core belief: {value_text(item.get('core_belief'))}. "
    f"Core belief probabilities: {format_probability_map(item.get('core_belief_probabilities'))}. "
    f"Distortion: {value_text(item.get('distortion'))}. "
    f"Distortion probabilities: {format_probability_map(item.get('distortion_probabilities'))}."
  )


def format_cognitive_sets(value: Any) -> str:
  if not isinstance(value, list) or not value:
    return "none"
  return " ".join(format_cognitive_set(item) for item in value)


def format_technique(value: Any) -> str:
  if value is None:
    return "none"
  if not isinstance(value, dict):
    return value_text(value)
  return (
    f"Name: {value_text(value.get('name'))}. "
    f"Target distortion: {value_text(value.get('target_distortion'))}. "
    f"Rationale: {value_text(value.get('rationale'))}."
  )


def format_techniques(value: Any) -> str:
  if not isinstance(value, list) or not value:
    return "none"
  return " ".join(f"Technique {idx}. {format_technique(item)}" for idx, item in enumerate(value, start=1))


def format_evaluations(value: Any) -> str:
  if not isinstance(value, list) or not value:
    return "none"
  return " ".join(f"Evaluation {idx}: {value_text(item)}." for idx, item in enumerate(value, start=1))


def format_tasks(value: Any) -> str:
  if not isinstance(value, list) or not value:
    return "none"
  rendered: list[str] = []
  for idx, item in enumerate(value, start=1):
    if isinstance(item, dict):
      rendered.append(
        f"Task {idx}. Title: {value_text(item.get('task_title'))}. "
        f"Description: {value_text(item.get('task_description'))}. "
        f"Rationale: {value_text(item.get('rationale'))}. "
        f"Expected outcomes: {value_text(item.get('expected_outcomes'))}."
      )
    else:
      rendered.append(f"Task {idx}. {value_text(item)}.")
  return " ".join(rendered)


def format_dialogue_message(message: dict[str, Any]) -> str:
  role = ensure_str(message.get("role")).strip().title() or "Unknown"
  idx = value_text(message.get("utterance_index"), empty="unknown")
  stage = ensure_str(message.get("stage")).strip()
  substage = ensure_str(message.get("substage")).strip()
  stage_text = f", {stage}/{substage}" if stage or substage else ""
  return f"{role} (utterance {idx}{stage_text}): {value_text(message.get('content'))}"


def recent_dialogue_text(last_utterances: list[Any], current_patient_message: dict[str, Any]) -> str:
  current_idx = current_patient_message.get("utterance_index")
  filtered: list[dict[str, Any]] = []
  skipped_current = False
  for message in last_utterances:
    if not isinstance(message, dict):
      continue
    is_current = (
      not skipped_current
      and ensure_str(message.get("role")).strip().lower() == "patient"
      and message.get("utterance_index") == current_idx
      and ensure_str(message.get("content")) == ensure_str(current_patient_message.get("content"))
    )
    if is_current:
      skipped_current = True
      continue
    filtered.append(message)

  selected = filtered[-10:]
  if not selected:
    return "none"
  return "\n".join(format_dialogue_message(message) for message in selected)


def format_session_context(
  context: dict[str, Any],
  last_utterances: list[Any],
  current_patient_message: dict[str, Any],
) -> str:
  return "\n".join(
    [
      (
        f"The current stage is: {value_text(context.get('current_stage'))}. "
        f"The current substage is: {value_text(context.get('current_substage'))}. "
        f"The reported patient mood was: {value_text(context.get('opening_patient_mood'), empty='not recorded')}. "
        f"The reported patient problem was: {value_text(context.get('opening_patient_problem'), empty='not recorded')}. "
        f"The predicted patient resistance was: {value_text(context.get('predicted_resistance'), empty='not available')}."
      ),
      "",
      f"Completed cognitive model sets: {format_cognitive_sets(context.get('cognitive_model_sets'))}.",
      f"Working cognitive model set: {format_cognitive_set(context.get('working_cognitive_set'))}.",
      f"Shelved cognitive model sets: {format_cognitive_sets(context.get('shelved_cognitive_sets'))}.",
      "",
      f"The cognitive gap plan was: {value_text(context.get('cognitive_gap_plan'))}.",
      f"The active cognitive target was: {value_text(context.get('active_cognitive_target'))}.",
      f"The active cognitive question was: {value_text(context.get('active_cognitive_question'))}.",
      "",
      f"Selected CBT techniques: {format_techniques(context.get('selected_techniques'))}.",
      f"The current CBT technique was: {format_technique(context.get('current_technique'))}.",
      f"The current technique plan was: {value_text(context.get('current_technique_plan'))}.",
      f"The expected current technique outcome was: {value_text(context.get('current_technique_expected_outcome'))}.",
      f"The current technique progress was: {value_text(context.get('current_technique_current_progress'))}.",
      "",
      f"Technique evaluations so far: {format_evaluations(context.get('technique_evaluations'))}.",
      f"Homework or agreed tasks: {format_tasks(context.get('tasks'))}.",
      f"The active therapy-interfering behavior to address was: {value_text(context.get('address_tib_name'))}.",
      "",
      "Recent dialogue:",
      recent_dialogue_text(last_utterances, current_patient_message),
    ]
  )


def last_patient_message(last_utterances: Any) -> dict[str, Any] | None:
  if not isinstance(last_utterances, list):
    return None
  for message in reversed(last_utterances):
    if not isinstance(message, dict):
      continue
    if ensure_str(message.get("role")).strip().lower() != "patient":
      continue
    if ensure_str(message.get("content")).strip():
      return message
  return None


def build_sample(turn: dict[str, Any], instruction: str, source_label: str) -> dict[str, str] | None:
  traces = turn.get("reasoning_traces")
  if not isinstance(traces, list):
    raise DatasetBuildError(f"{source_label}: turn reasoning_traces must be a JSON list.")
  if not traces:
    return None

  last_utterances = turn.get("last_utterances")
  patient_message = last_patient_message(last_utterances)
  if patient_message is None:
    return None

  context = require_dict(turn.get("context_before_therapist_turn"), f"{source_label}: context_before_therapist_turn")
  therapist_utterance = ensure_str(turn.get("therapist_utterance")).strip()
  if not therapist_utterance:
    raise DatasetBuildError(f"{source_label}: turn therapist_utterance must be non-empty.")

  trace_lines = []
  for trace in traces:
    trace_lines.append(trace_reasoning_text(require_dict(trace, f"{source_label}: reasoning trace")))
  trace_text = "\n".join(trace_lines)

  context_text = format_session_context(context, require_list(last_utterances, f"{source_label}: last_utterances"), patient_message)
  input_text = f"<context>\n{context_text}\n</context>\n{value_text(patient_message.get('content'))}"
  output_text = f"<think>{trace_text}</think>{therapist_utterance}"
  return {
    "instruction": instruction,
    "input": input_text,
    "output": output_text,
  }


def load_session(path: Path) -> dict[str, Any]:
  try:
    with path.open("r", encoding="utf-8") as f:
      payload = json.load(f)
  except json.JSONDecodeError as exc:
    raise DatasetBuildError(f"{path}: malformed JSON: {exc}") from exc
  except OSError as exc:
    raise DatasetBuildError(f"{path}: failed to read file: {exc}") from exc
  return require_dict(payload, f"{path}: root")


def build_samples_for_file(path: Path, system_prompt: str) -> list[dict[str, str]]:
  payload = load_session(path)
  profile = require_dict(payload.get("patient_profile"), f"{path}: patient_profile")
  turns = require_list(payload.get("turns"), f"{path}: turns")
  instruction = render_instruction(system_prompt, profile)

  samples: list[dict[str, str]] = []
  for idx, raw_turn in enumerate(turns, start=1):
    turn = require_dict(raw_turn, f"{path}: turns[{idx}]")
    sample = build_sample(turn, instruction, f"{path}: turns[{idx}]")
    if sample is not None:
      samples.append(sample)
  return samples


def build_dataset(input_dir: Path, output_file: Path, root: Path | None = None) -> int:
  if not input_dir.is_dir():
    raise DatasetBuildError(f"--input-dir must be a directory: {input_dir}")
  root = root or project_root()
  system_prompt = load_therapist_system_prompt(root)
  input_files = sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json")

  samples: list[dict[str, str]] = []
  for input_file in input_files:
    samples.extend(build_samples_for_file(input_file, system_prompt))

  output_file.parent.mkdir(parents=True, exist_ok=True)
  with output_file.open("w", encoding="utf-8", newline="\n") as f:
    for sample in samples:
      f.write(json.dumps(sample, ensure_ascii=False))
      f.write("\n")
  return len(samples)


def main() -> int:
  parser = argparse.ArgumentParser(
    description="Create a reasoning instruction-tuning JSONL dataset from reconstructed therapy sessions."
  )
  parser.add_argument("--input-dir", required=True, help="Directory containing reconstructed session JSON files.")
  parser.add_argument("--output-file", required=True, help="Output JSONL file to write.")
  args = parser.parse_args()

  try:
    count = build_dataset(Path(args.input_dir), Path(args.output_file))
  except DatasetBuildError as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1

  print(f"Wrote {count} samples to {args.output_file}", file=sys.stderr)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
