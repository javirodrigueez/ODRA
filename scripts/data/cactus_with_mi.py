import json
import argparse
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv()

# --- Knowledge Graph and Definitions ---
TOPIC_GRAPH = {
    "Health": {"Parent": [], "Children": ["Mental Disorders", "Diseases", "Sexual Health", "Fitness", "Health Care", "Workplace Wellness"]},
    "Interpersonal Relationships": {"Parent": [], "Children": ["Family", "Parenting", "Workplace Relationships"]},
    "Law": {"Parent": [], "Children": ["Criminal Law", "Child Custody", "Traffic Ticket"]},
    "Economy": {"Parent": [], "Children": ["Employment", "Personal Finance"]},
    "Education": {"Parent": [], "Children": ["Student Affairs", "Exam", "Academic Achievement"]},
    "Mental Disorders": {"Parent": ["Health"], "Children": ["Depression", "Chronodisruption", "Anxiety Disorders", "Cognitive Decline"]},
    "Diseases": {"Parent": ["Health"], "Children": ["Infection", "Hypertension", "Flu", "Inflammation", "Liver Disease", "Lung Cancer", "COPD", "Asthma", "Stroke", "Diabetes"]},
    "Sexual Health": {"Parent": ["Health"], "Children": ["Birth Defects", "Maternal Health", "Preterm Birth", "Miscarriage", "Safe Sex"]},
    "Fitness": {"Parent": ["Health"], "Children": ["Physical Activity", "Sport", "Physical Fitness", "Strength", "Flexibility", "Endurance"]},
    "Health Care": {"Parent": ["Health"], "Children": ["Dentistry", "Caregiver Burden", "Independent Living", "Human Appearance", "Child Care", "Maternal Health"]},
    "Parenting": {"Parent": ["Interpersonal Relationships"], "Children": ["Role Model", "Child Development", "Paternal Bond", "Child Care", "Habituation", "Child Custody"]},
    "Family": {"Parent": ["Interpersonal Relationships"], "Children": ["Family Estrangement", "Family Disruption", "Divorce", "Child Custody"]},
    "Criminal Law": {"Parent": ["Law"], "Children": ["Arrest", "Imprisonment", "Complaint"]},
    "Personal Finance": {"Parent": ["Economy"], "Children": ["Cost of Living", "Personal Budget", "Debt", "Income Deficit"]},
    "Employment": {"Parent": ["Economy"], "Children": ["Productivity", "Absenteeism", "Workplace Relationships", "Career Break", "Career Assessment", "Absence Rate", "Salary", "Workplace Wellness", "Workplace Incivility"]},
    "Student Affairs": {"Parent": ["Education"], "Children": ["Attendance", "Suspension", "Exam", "Scholarship"]},
    "Cognitive Decline": {"Parent": ["Mental Disorders"], "Children": []},
    "Chronodisruption": {"Parent": ["Mental Disorders"], "Children": []},
    "Depression": {"Parent": ["Mental Disorders"], "Children": []},
    "Anxiety Disorders": {"Parent": ["Mental Disorders"], "Children": []},
    "Infection": {"Parent": ["Diseases"], "Children": []},
    "Diabetes": {"Parent": ["Diseases"], "Children": []},
    "Hypertension": {"Parent": ["Diseases"], "Children": []},
    "Liver Disease": {"Parent": ["Diseases"], "Children": []},
    "COPD": {"Parent": ["Diseases"], "Children": []},
    "Asthma": {"Parent": ["Diseases"], "Children": []},
    "Flu": {"Parent": ["Diseases"], "Children": []},
    "Lung Cancer": {"Parent": ["Diseases"], "Children": []},
    "Stroke": {"Parent": ["Diseases"], "Children": []},
    "Inflammation": {"Parent": ["Diseases"], "Children": []},
    "Birth Defects": {"Parent": ["Sexual Health"], "Children": []},
    "Preterm Birth": {"Parent": ["Sexual Health"], "Children": []},
    "Safe Sex": {"Parent": ["Sexual Health"], "Children": []},
    "Miscarriage": {"Parent": ["Sexual Health"], "Children": []},
    "Maternal Health": {"Parent": ["Sexual Health", "Health Care"], "Children": []},
    "Strength": {"Parent": ["Fitness"], "Children": []},
    "Physical Fitness": {"Parent": ["Fitness"], "Children": []},
    "Flexibility": {"Parent": ["Fitness"], "Children": []},
    "Physical Activity": {"Parent": ["Fitness"], "Children": []},
    "Sport": {"Parent": ["Fitness"], "Children": []},
    "Endurance": {"Parent": ["Fitness"], "Children": []},
    "Caregiver Burden": {"Parent": ["Health Care"], "Children": []},
    "Independent Living": {"Parent": ["Health Care"], "Children": []},
    "Human Appearance": {"Parent": ["Health Care"], "Children": []},
    "Dentistry": {"Parent": ["Health Care"], "Children": []},
    "Child Care": {"Parent": ["Health Care", "Parenting"], "Children": []},
    "Paternal Bond": {"Parent": ["Parenting"], "Children": []},
    "Role Model": {"Parent": ["Parenting"], "Children": []},
    "Habituation": {"Parent": ["Parenting"], "Children": []},
    "Child Development": {"Parent": ["Parenting"], "Children": []},
    "Family Disruption": {"Parent": ["Family"], "Children": []},
    "Divorce": {"Parent": ["Family"], "Children": []},
    "Family Estrangement": {"Parent": ["Family"], "Children": []},
    "Child Custody": {"Parent": ["Family", "Parenting", "Law"], "Children": []},
    "Traffic Ticket": {"Parent": ["Law"], "Children": []},
    "Imprisonment": {"Parent": ["Criminal Law"], "Children": []},
    "Complaint": {"Parent": ["Criminal Law"], "Children": []},
    "Arrest": {"Parent": ["Criminal Law"], "Children": []},
    "Personal Budget": {"Parent": ["Personal Finance"], "Children": []},
    "Debt": {"Parent": ["Personal Finance"], "Children": []},
    "Cost of Living": {"Parent": ["Personal Finance"], "Children": []},
    "Income Deficit": {"Parent": ["Personal Finance"], "Children": []},
    "Workplace Wellness": {"Parent": ["Health", "Employment"], "Children": []},
    "Workplace Incivility": {"Parent": ["Employment"], "Children": []},
    "Workplace Relationships": {"Parent": ["Employment", "Interpersonal Relationships"], "Children": []},
    "Salary": {"Parent": ["Employment"], "Children": []},
    "Career Break": {"Parent": ["Employment"], "Children": []},
    "Absence Rate": {"Parent": ["Employment"], "Children": []},
    "Career Assessment": {"Parent": ["Employment"], "Children": []},
    "Absenteeism": {"Parent": ["Employment"], "Children": []},
    "Productivity": {"Parent": ["Employment"], "Children": []},
    "Academic Achievement": {"Parent": ["Education"], "Children": []},
    "Exam": {"Parent": ["Education"], "Children": []},
    "Suspension": {"Parent": ["Student Affairs"], "Children": []},
    "Attendance": {"Parent": ["Student Affairs"], "Children": []},
    "Scholarship": {"Parent": ["Student Affairs"], "Children": []}
}

# Extract leaf nodes (fine-grained topics)
FINE_GRAINED_TOPICS = [
    topic for topic, relations in TOPIC_GRAPH.items() 
    if len(relations.get("Children", [])) == 0
]

def generate_motivation_prompt(patient_data_str: str) -> str:
    return f"""
<persona>
{patient_data_str}
</persona>

<topics>
{json.dumps(FINE_GRAINED_TOPICS, indent=2)}
</topics>

You are CBT patient which persona is defined in <persona>. You may have resistance or rejection to therapy. Based on the patient's profile, identify the single most effective motivation topic that a counselor could use to bypass this resistance and encourage your behavioral change. You MUST select exactly ONE topic from the fine-grained topics list defined in <topics>.

Output your response strictly as a JSON object with a single key "motivation_topic". Do not include markdown formatting or extra text.
"""

def process_patient_seeds(input_path: str, output_path: str):
    if not os.environ.get("OR_API_KEY"):
        print("Error: OR_API_KEY environment variable is missing.")
        sys.exit(1)

    # Initialize standard OpenAI SDK pointed at OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OR_API_KEY"),
    )

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            seeds = json.load(f)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)

    print(f"Loaded {len(seeds)} patient seeds. Beginning LLM motivation topic assignment...")

    found_names = dict()  # Cache to reuse motivation topics for identical names
  
    for i, seed in enumerate(tqdm(seeds)):
        patient_str = json.dumps(seed, indent=2)
        name = seed.get("personal_info").get("name")

        # Reuse topic from previous identical name
        if name in found_names:
            seed["motivation_topic"] = found_names[name]  
            continue
        
        prompt = generate_motivation_prompt(patient_str)
        
        try:
            # Requesting specific model execution via OpenRouter provider parameters
            response = client.chat.completions.create(
                model="deepseek/deepseek-v3.2",
                extra_body={
                    "provider": {
                        "order": ["DeepSeek"]
                    }
                },
                messages=[
                    {"role": "system", "content": "You are a robust JSON-only output agent."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2
            )

            result = json.loads(str(response.choices[0].message.content))
            
            # Basic validation
            assigned_topic = result.get("motivation_topic")
            if assigned_topic not in FINE_GRAINED_TOPICS:
                print(f"Warning: Model output '{assigned_topic}' for seed idx {i} is not in the fine-grained list. Proceeding with fallback.")
            
            seed["motivation_topic"] = assigned_topic
            found_names[name] = assigned_topic

        except Exception as e:
            print(f"API Error processing seed idx {i}: {e}")
            seed["motivation_topic"] = None

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(seeds, f, indent=2)
    
    print(f"Successfully processed and saved to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assign fine-grained motivation topics to CBT patient seeds using DeepSeek via OpenRouter.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input JSON file containing patient seeds.")
    parser.add_argument('--output', type=str, required=True, help="Path to save the modified JSON output.")
    args = parser.parse_args()

    process_patient_seeds(args.input, args.output)