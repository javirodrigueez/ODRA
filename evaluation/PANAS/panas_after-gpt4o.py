from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
import json
import argparse
import os
from openai import AzureOpenAI, OpenAI
import tiktoken
import re
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

FEELINGS = [
    'interested', 'excited', 'strong', 'enthusiastic', 'proud',
    'alert', 'inspired', 'determined', 'attentive', 'active',
    'distressed', 'upset', 'guilty', 'scared', 'hostile',
    'irritable', 'ashamed', 'nervous', 'jittery', 'afraid',
]

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Function to count tokens
def num_tokens_from_message(messages, model="gpt-4o"):
    enc = tiktoken.encoding_for_model("gpt-4o")
    tokens_per_message = 3
    total_tokens = 0

    for message in messages:
        total_tokens += tokens_per_message + len(enc.encode(message["content"]))
    
    total_tokens += 3
    return total_tokens

def calculate_cost(input_message, output):
    input_cost = (num_tokens_from_message(input_message)*(2.5))/1000000
    output_cost = (num_tokens_from_string(output)*(10))/1000000
    return input_cost+output_cost

def generate_history(history):
    history_text = '\n'.join(
            [
                f"{message['role'].capitalize()}: {message['content']}"
                for message in history
            ]
    )

    return history_text

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def _parse_score_line(line: str) -> tuple[str, str, int] | None:
    stripped = line.strip()
    if not stripped:
        return None

    parts = [part.strip() for part in stripped.split(",")]
    if len(parts) >= 3:
        emotion = re.sub(r'[^a-zA-Z0-9]', '', parts[0]).lower()
        explanation = ",".join(parts[1:-1]).strip()
        score_str = re.sub(r'[^a-zA-Z0-9]', '', parts[-1])
    else:
        txt_list_words = stripped.split()
        if len(txt_list_words) < 2:
            return None
        emotion = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[0]).lower()
        explanation = ""
        score_str = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[-1])

    if emotion not in FEELINGS or not score_str.isdigit():
        return None

    return emotion, explanation, int(score_str)

def score(response, max_iter, include_explanations: bool = False):
    score_dict = {feeling: [] for feeling in FEELINGS}
    explanations: list[dict[str, Any]] = []
    for i in range(max_iter):
        txt = response.choices[i].message.content or ""
        txt_list = txt.split('\n')
        for line in txt_list:
            parsed_line = _parse_score_line(line)
            if parsed_line is None:
                continue
            emotion, explanation, value = parsed_line
            score_dict[emotion].append(value)
            if include_explanations:
                explanations.append(
                    {
                        "emotion": emotion,
                        "explanation": explanation,
                        "score": value,
                    }
                )
    if include_explanations:
        score_dict["explanations"] = explanations
    return score_dict

def save_as_json(dictionary, filename):
    """Save dictionary to a JSON file"""
    with open(filename, 'w') as file:
        json.dump(dictionary, file, indent=4)
    print(f"Dictionary saved to {filename}")

def format_persona(info) -> str:
  lines = []
  for key, val in info.items():
    label = key.replace("_", " ").title()
    lines.append(f"{label}: {val}")
  return "\n".join(lines)

def format_more_info(info) -> str:
  sections = [
    ("Past History", info.get("past_history", "")),
    ("Academic/Occupational Functioning", info.get("academic_occupational_functioning", "")),
    ("Social Support System", info.get("social_support_system", "")),
  ]
  return "\n".join(f"{title}:\n{content}" for title, content in sections if content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate PANAS results")
    parser.add_argument("-i","--input_dir", type=str, default=".",
                        help="Directory to read the sessions")
    parser.add_argument("-p","--presession_dir", type=str, default=".",
                        help="Directory to read the presession questionnaires")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter","--max_iter", type=int, default=3,
                        help="Number of times GPT-4o is run for scoring a single client.")
    parser.add_argument(
        "--include_explanations",
        action="store_true",
        help="Include explanation records in the output JSON and use the explanation prompt format.",
    )

    args = parser.parse_args()
    
    model_name = "gpt-4o"
    #model_name = "gpt-4o-mini"
    #model_name = "deepseek/deepseek-v3.2"
    
    
    evaluator = OpenAI(
        api_key=os.getenv("GPT_API_KEY") or "", 
        base_url="https://api.openai.com/v1"
        #base_url="https://openrouter.ai/api/v1"
    )

    # with open("../../dataset/data.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def format_presession_questionnaire(file_path):
      """
      Parses a JSON file to extract top-level emotional scores.
      """
      try:
          with open(file_path, 'r') as f:
              data = json.load(f)
          
          # Filter out non-score keys (explanations, cost, attitude)
          output = []
          for key, value in data.items():
              if isinstance(value, list) and len(value) > 0 and isinstance(value[0], int):
                  # Format each entry as "feeling": score
                  output.append(f'{key}: {value[0]}')
          
          return "\n".join(output)

      except (FileNotFoundError, json.JSONDecodeError) as e:
          return f"Error processing file: {e}"
        

    for filename in os.listdir(args.input_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(args.input_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            intake_form = f"{format_persona(json_data["patient_profile"]["personal_info"])}\n{format_more_info(json_data["patient_profile"]["more_information"])}"
            presession_questionnaire = format_presession_questionnaire(os.path.join(args.presession_dir, filename))
            total_cost = 0
            prompt_name = "panas_after.txt" if args.include_explanations else "../prompts_score/panas_after.txt"
            prompt_text = load_prompt(prompt_name)
            prompt_template = PromptTemplate(
                                input_variables=["intake_form","conversation", "presession_questionnaire"],
                                template=prompt_text)
            prompt = prompt_template.format(
                        intake_form = intake_form,
                        conversation = generate_history(json_data["messages"]),
                        presession_questionnaire = presession_questionnaire
                    )
            messages: Any = [{'role': 'user', 'content': prompt}]
            response = evaluator.chat.completions.create(
                        messages=messages,
                        temperature=0,
                        model=model_name,
                        n=args.max_iter
                        )
            score_dict: Any = score(response, args.max_iter, include_explanations=args.include_explanations)
            score_dict['cost'] = 0
            score_dict['attitude'] = json_data["patient_profile"]["attitude"]
            for i in range(args.max_iter):
                score_dict['cost'] = score_dict['cost'] + calculate_cost(messages,response.choices[i].message.content)
            save_as_json(score_dict, os.path.join(args.output_dir, filename))
