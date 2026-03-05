from graph import TherapyGraphBuilder
import argparse
import json
import os
from omegaconf import OmegaConf
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

def main(cfg_path: str):
  load_dotenv()

  cfg = OmegaConf.load(cfg_path)

  profiles_path = cfg.patient.profiles
  max_turns = cfg.env.max_turns
  output_dir = cfg.env.get('output_dir', None)
  verbose = cfg.env.get('verbose', False)

  # Load profiles from JSONL
  profiles = []
  with open(profiles_path, 'r') as f:
    for line in f:
      line = line.strip()
      if line:
        profiles.append(json.loads(line))

  print(f'Loaded {len(profiles)} profiles from {profiles_path}\n')

  for i, profile in enumerate(profiles):
    print(f'{"="*60}')
    print(f'Simulation {i+1}/{len(profiles)} — idx: {profile["idx"]}')
    print(f'Topic: {profile["topic"]}, Behavior: {profile["Behavior"]}')
    print(f'{"="*60}')

    result = run_simulation(cfg_path, profile, max_turns, output_dir, verbose)
    print(f'  Total turns: {result["current_turn"]}\n')

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Run CBT therapy simulation')
  parser.add_argument('--cfg', type=str, default='configs/bedrock.yaml', help='YAML configuration file.')
  args = parser.parse_args()
  main(args.cfg)