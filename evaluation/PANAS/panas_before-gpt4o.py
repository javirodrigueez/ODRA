import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import json
import argparse
from openai import AzureOpenAI, OpenAI
import tiktoken
import re
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

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
    input_cost = (num_tokens_from_message(input_message)*(5))/1000000
    output_cost = (num_tokens_from_string(output)*(20))/1000000
    return input_cost+output_cost

def load_prompt(file_name):
    base_dir = "prompts/"
    file_path = base_dir + file_name
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def score(response, max_iter):
    score_dict = {}
    feelings = ['interested', 'excited', 'strong', 'enthusiastic', 'proud', 'alert', 'inspired', 'determined', 'attentive', 'active', 'distressed', 'upset', 'guilty', 'scared', 'hostile', 'irritable', 'ashamed', 'nervous', 'jittery', 'afraid']
    for i in range(len(feelings)):
        score_dict[feelings[i]] = []
    for i in range(max_iter):
        txt = response.choices[i].message.content
        txt_list = txt.split('\n')
        for j in range(len(txt_list)):
            txt_list_words = txt_list[j].split()
            feel = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[0]).lower()
            score_str = re.sub(r'[^a-zA-Z0-9]', '', txt_list_words[-1])
            score = int(score_str)
            score_dict[feel].append(score)
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
    parser.add_argument("-i","--input_file", type=str, default=".",
                        help="Path to the input JSON file.")
    parser.add_argument("-o","--output_dir", type=str, default=".",
                        help="Directory to to save the results.")
    parser.add_argument("-m_iter","--max_iter", type=int, default=3,
                        help="Number of times GPT-4o is run for scoring a single client.")

    args = parser.parse_args()
    
    model_name = "gpt-4o"
      
    evaluator = OpenAI(
        api_key=os.getenv("GPT_API_KEY") or "", 
        base_url="https://api.openai.com/v1"
    )

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for number in range(len(data)):
        total_cost = 0
        prompt_text = load_prompt("panas_before.txt")
        prompt_template = PromptTemplate(
                            input_variables=["intake_form"],
                            template=prompt_text)
        prompt = prompt_template.format(
                    intake_form = f"{format_persona(data[number]['personal_info'])}\n{format_more_info(data[number]['more_information'])}"
                )
        messages: Any = [{'role': 'user', 'content': prompt}]
        response = evaluator.chat.completions.create(
                    messages=messages,
                    temperature=0,
                    model=model_name,
                    n=args.max_iter
                    )
        score_dict = score(response, args.max_iter)
        score_dict['cost'] = 0
        score_dict['attitude'] = data[number]['attitude']
        for i in range(args.max_iter):
            score_dict['cost'] = score_dict['cost'] + calculate_cost(messages,response.choices[i].message.content)
        save_as_json(score_dict, args.output_dir + '/simulation_' + str(number+1) + '.json')
