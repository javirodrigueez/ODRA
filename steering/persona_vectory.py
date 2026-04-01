import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_persona_vector(model, tokenizer, layer_idx, resistant_prompts, cooperative_prompts):
    """
    Extracts the persona vector by calculating the mean difference in activations.
    """
    def get_mean_activation(prompts):
        activations = []
        def hook_fn(module, input, output):
            # output[0] contains the hidden states
            activations.append(output[0].detach().mean(dim=1)) # Mean over sequence length
            
        handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                model(**inputs)
                
        handle.remove()
        return torch.cat(activations, dim=0).mean(dim=0)

    print("Extracting activations for resistant responses...")
    mean_resistant = get_mean_activation(resistant_prompts)
    
    print("Extracting activations for cooperative responses...")
    mean_cooperative = get_mean_activation(cooperative_prompts)
    
    persona_vector = mean_resistant - mean_cooperative
    # Normalize vector to ensure consistent scaling
    return persona_vector / torch.norm(persona_vector)

def generate_steered_response(model, tokenizer, prompt, persona_vector, layer_idx, alpha):
    """
    Generates a response while steering the specified layer with the persona vector.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def steering_hook(module, input, output):
        # Steer the hidden states: h_l <- h_l + \alpha * v_l
        # Modifying output[0] which is the hidden state tensor
        steered_hidden = output[0] + (alpha * persona_vector.to(output[0].device))
        return (steered_hidden,) + output[1:]
        
    handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7)
        
    handle.remove()
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Steer patient resistance in LLM using Persona Vectors.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--layer", type=int, default=16, help="Transformer layer index for steering")
    parser.add_argument("--r_value", type=float, default=0.8, help="Patient resistance value [0, 1]")
    parser.add_argument("--max_alpha", type=float, default=2.5, help="Maximum steering coefficient multiplier")
    args = parser.parse_args()

    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16, device_map="auto")

    # Toy data - In practice, substitute with your diverse synthetic generation set
    resistant_prompts = ["You are a highly resistant therapy patient. Therapist: How was your week?"]
    cooperative_prompts = ["You are a cooperative therapy patient. Therapist: How was your week?"]

    vector = extract_persona_vector(model, tokenizer, args.layer, resistant_prompts, cooperative_prompts)
    
    # Map R value [0,1] to alpha steering coefficient
    alpha = args.r_value * args.max_alpha
    print(f"Mapped R-value {args.r_value} to Steering Coefficient (Alpha): {alpha}")

    test_prompt = "Therapist: Have you had a chance to reflect on the homework we discussed last session?"
    
    print("\n--- Generating Steered Response ---")
    response = generate_steered_response(model, tokenizer, test_prompt, vector, args.layer, alpha)
    print(f"Patient: {response}")

if __name__ == "__main__":
    main()