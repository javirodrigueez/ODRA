import json
import argparse
import sys

def extract_messages(file_path, out_path):
    try:
        # Load the JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        out_file = open(out_path, 'w', encoding='utf-8') if out_path else sys.stdout
        
        # Access the 'messages' list
        messages = data.get('messages', [])
        
        if not messages:
            print(f"No messages found in {file_path}")
            return

        # Iterate and format: {role}:{content}
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            out_file.write(f"-->{role.capitalize()}:{content}\n") if out_file else None
        out_file.close() if out_file else None

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from '{file_path}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract role and content from a JSON conversation file.")
    
    # Adding the input argument
    parser.add_argument('--input', help="Path to the initial JSON file")
    parser.add_argument('--output', help="Path to the output file")

    args = parser.parse_args()
    
    extract_messages(args.input, args.output)

if __name__ == "__main__":
    main()