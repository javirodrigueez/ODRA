from graph import TherapyGraphBuilder
import argparse
import json
import os
from dotenv import load_dotenv

def run_simulation(cfg_path: str, profile: dict, max_turns: int, output_dir: str = None, verbose: bool = False):
  """Run a single simulation for a given profile."""
  builder = TherapyGraphBuilder(cfg_path, profile)
  chain = builder.createChain()

  if verbose:
    print()
    last_msg_count = 0
    for chunk in chain.stream({'max_turns': max_turns}, stream_mode="values"):
      messages = chunk.get('messages', [])
      # Print only new messages since last chunk
      for msg in messages[last_msg_count:]:
        print(f'  {msg.content}')
      last_msg_count = len(messages)
    result = chunk  # last chunk is the final state
    print()
  else:
    result = chain.invoke({'max_turns': max_turns})

  if output_dir:
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'simulation_{profile["idx"]}.txt')
    with open(output_file, 'w') as f:
      f.write(f'Profile idx: {profile["idx"]}\n')
      f.write(f'Topic: {profile["topic"]}\n')
      f.write(f'Behavior: {profile["Behavior"]}\n')
      f.write('\n' + '='*60 + '\n')
      f.write('SESSION TRANSCRIPT\n')
      f.write('='*60 + '\n')
      for message in result['messages']:
        f.write('--> ' + message.content + '\n\n')
    print(f'  Saved to: {output_file}')

  return result

def main(args):
  load_dotenv()

  # Load profiles from JSONL
  profiles = []
  with open(args.profiles, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        profiles.append(json.loads(line))

  print(f'Loaded {len(profiles)} profiles from {args.profiles}\n')

  for i, profile in enumerate(profiles):
    print(f'{"="*60}')
    print(f'Simulation {i+1}/{len(profiles)} — idx: {profile["idx"]}')
    print(f'Topic: {profile["topic"]}, Behavior: {profile["Behavior"]}')
    print(f'{"="*60}')

    result = run_simulation(args.cfg, profile, args.max_turns, args.output_dir, args.verbose)
    print(f'  Total turns: {result["current_turn"]}\n')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run CBT therapy simulation')
  parser.add_argument('--cfg', type=str, default='configs/bedrock.yaml', help='LLM configuration file.')
  parser.add_argument('--profiles', type=str, default='data/profiles_test.jsonl', help='JSONL file with patient profiles.')
  parser.add_argument('--max-turns', type=int, default=10, help='Maximum number of turns per simulation.')
  parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save session transcripts.')
  parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
  args = parser.parse_args()
  main(args)