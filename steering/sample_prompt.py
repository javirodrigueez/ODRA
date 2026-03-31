from openai import OpenAI

# Initialize the client to point to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="local-no-key-required"  # A placeholder is usually required
)

def send_local_prompt(user_input):
    try:
        # Change "model-name" to whatever your local model is named
        # (e.g., "llama3", "mistral", or "gpt-3.5-turbo" if aliased)
        response = client.chat.completions.create(
            model="/synther/steering/models/Qwen3.5-35B-FP8/", 
            messages=[
                {"role": "system", "content": "You are a helpful local assistant."},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7
        )
        
        print(f"Model Response:\n {response.choices[0].message.content}\n\n")
        print(f"Full Response:\n {response}\n\n")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    prompt = "Differences between Qwen and other language models."
    send_local_prompt(prompt)