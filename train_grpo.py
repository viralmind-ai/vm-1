"""
Computer-use reward functions inspired by DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
The format reward and accuracy reward functions have been adapted for coordinate-based tasks.

These functions can be used to train a visual model to reason over computer-use tasks.
"""
# train_grpo.py
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

import re
import ast

# note: only the 'prompt' column gets used
dataset = load_dataset("trl-lib/tldr", split="train")
# dataset = load_dataset("rootsautomation/ScreenSpot", split="train")
# TODO: publish viralmind/gym on huggingface

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format.
    Inspired by the format reward from DeepSeek-R1 paper."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def coordinate_reward_func(completions, ground_truth, **kwargs):
    """Reward function that checks if predicted coordinates are within the ground truth bounding box.
    Inspired by the accuracy reward from DeepSeek-R1 paper."""
    pattern = r'<coordinates>(\d+\.?\d*),\s*(\d+\.?\d*)</coordinates>'
    def check_coords(completion, bbox_str):
        try:
            match = re.search(pattern, completion)
            x, y = map(float, match.groups()) if match else (0, 0)
            x1, y1, x2, y2 = ast.literal_eval(bbox_str)
            return 1.0 if (x1 <= x <= x2 and y1 <= y <= y2) else 0.0
        except:
            return 0.0
    return [check_coords(c, gt) for c, gt in zip(completions, ground_truth)]


# Test both reward functions
prompts = ["Click the button", "Click the image"]
completions = [
    "<think>The button is in the center</think><answer>Click at <coordinates>150.0, 200.0</coordinates></answer>",
    "Click at <coordinates>50.0, 50.0</coordinates>"  # Wrong format
]
ground_truth = ["[100.0, 100.0, 300.0, 300.0]", "[0.0, 0.0, 100.0, 100.0]"]

print("Format rewards:", format_reward_func(completions))
print("Coordinate rewards:", coordinate_reward_func(completions, ground_truth))
# Output should be:
# Format rewards: [1.0, 0.0]
# Coordinate rewards: [1.0, 1.0]

# Train a LLM using reinforcement learning
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[format_reward_func, coordinate_reward_func],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()