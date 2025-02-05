import random
import string
import numpy as np
from dataclasses import dataclass
from typing import List, Set
import imageio
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor  # Hugging Face tokenizer

# === Interaction Frame Definition & Encoding (Example) ===
@dataclass
class InteractionFrame:
    mouse_x: float
    mouse_y: float
    mouse_click: bool
    keys_pressed: Set[str]
    timestamp: float

def encode_for_tokenizer(frames: List[InteractionFrame], width: int, height: int) -> np.ndarray:
    """
    Convert a list of InteractionFrame objects into a normalized NumPy array
    suitable for the FAST tokenizer.
    Encoding includes normalized x, normalized y, click,
    and a one-hot encoding for each of the 26 lowercase letters.
    """
    all_possible_keys = list(string.ascii_lowercase)  # 26 letters
    action_dim = 3 + len(all_possible_keys)  # x, y, click, and one-hot for keys
    actions = np.zeros((len(frames), action_dim), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        # Normalize mouse x/y to [-1, 1]
        actions[i, 0] = (frame.mouse_x / width) * 2 - 1
        actions[i, 1] = (frame.mouse_y / height) * 2 - 1
        actions[i, 2] = float(frame.mouse_click)
        
        # One-hot encode the keys pressed.
        for key in frame.keys_pressed:
            if key in all_possible_keys:
                key_idx = all_possible_keys.index(key) + 3  # offset by 3 for x, y, click
                actions[i, key_idx] = 1.0
                
    return actions

# === Random Trajectory Generator (with target points + fuzz) ===
def generate_random_trajectory(num_frames: int, width: int, height: int, segment_length: int = 10) -> List[InteractionFrame]:
    """
    Generate a random sequence of InteractionFrames.
    The mouse starts at the center, then for each segment it picks a new random
    target point. Over the course of the segment the mouse moves smoothly from
    the current point to the target while adding some random jitter ("fuzz") to make
    the path look more realistic.
    
    Additionally, a mouse click occurs with ~10% chance per frame,
    and with ~5% chance a random lowercase letter is pressed on that frame.
    """
    frames = []
    # Start at the center of the canvas.
    current_x, current_y = width / 2, height / 2
    frame_idx = 0

    while frame_idx < num_frames:
        # Pick a new random target point.
        target_x = random.uniform(0, width)
        target_y = random.uniform(0, height)
        
        # Compute incremental steps to move smoothly to the target.
        dx = (target_x - current_x) / segment_length
        dy = (target_y - current_y) / segment_length
        
        for i in range(segment_length):
            if frame_idx >= num_frames:
                break
            # Add random fuzz (jitter) to each step.
            fuzz_x = random.gauss(0, 3)  # standard deviation of 3 pixels
            fuzz_y = random.gauss(0, 3)
            x = current_x + dx * i + fuzz_x
            y = current_y + dy * i + fuzz_y
            # Clamp to the canvas boundaries.
            x = min(max(x, 0), width)
            y = min(max(y, 0), height)
            
            # Randomly simulate a mouse click (~10% chance).
            click = random.random() < 0.1
            
            # Randomly simulate a key press (~15% chance).
            keys = set()
            if random.random() < 0.15:
                keys.add(random.choice(string.ascii_lowercase))
                
            timestamp = frame_idx * 0.1  # For example, 0.1 sec per frame.
            frame = InteractionFrame(mouse_x=x, mouse_y=y, mouse_click=click, keys_pressed=keys, timestamp=timestamp)
            frames.append(frame)
            frame_idx += 1
        
        # Set the current position to the target for the next segment.
        current_x, current_y = target_x, target_y
        
    return frames

# === Decoding an Action Vector Back to an InteractionFrame ===
def decode_action_vector(action: np.ndarray, width: int, height: int) -> InteractionFrame:
    """
    Convert a normalized action vector (obtained from tokenizer decode)
    back into an InteractionFrame.
    (Note: The original timestamp isn’t preserved, so it’s assigned later.)
    """
    all_possible_keys = list(string.ascii_lowercase)
    # Convert normalized x, y back to absolute coordinates.
    x_norm, y_norm = action[0], action[1]
    x = ((x_norm + 1) / 2) * width
    y = ((y_norm + 1) / 2) * height
    click = bool(round(action[2]))
    keys = set()
    for idx, key in enumerate(all_possible_keys):
        if action[3 + idx] > 0.5:
            keys.add(key)
    return InteractionFrame(mouse_x=x, mouse_y=y, mouse_click=click, keys_pressed=keys, timestamp=0.0)

# === GIF Rendering Function ===
def create_gif_from_frames(frames: List[InteractionFrame], filename: str, width: int, height: int, fps: int = 10):
    """
    Create a GIF animation visualizing the trajectory. For each frame:
      - Draw the accumulated trajectory as a blue line.
      - Draw the current mouse position as a red circle.
      - Overlay the keys pressed on that frame.
      - Display the accumulated typed text in the lower corner.
    """
    images = []
    accumulated_text = ""
    trajectory_points = []
    
    # Attempt to load a font (using a truetype font if available).
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font = ImageFont.load_default()
    
    for frame in frames:
        # Create a blank white image.
        img = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(img)
        
        # Append current point and draw the trajectory so far.
        trajectory_points.append((frame.mouse_x, frame.mouse_y))
        if len(trajectory_points) > 1:
            draw.line(trajectory_points, fill="blue", width=2)
        
        # Draw the mouse cursor as a red circle.
        r = 5  # radius of the circle
        draw.ellipse(
            (frame.mouse_x - r, frame.mouse_y - r, frame.mouse_x + r, frame.mouse_y + r),
            fill="red"
        )
        
        # Draw the keys pressed in the current frame.
        keys_text = "Keys: " + (", ".join(sorted(frame.keys_pressed)) if frame.keys_pressed else "None")
        draw.text((10, 10), keys_text, fill="black", font=font)
        
        # Update and display the accumulated text.
        if frame.keys_pressed:
            for key in sorted(frame.keys_pressed):
                accumulated_text += key
        draw.text((10, height - 30), "Typed: " + accumulated_text, fill="black", font=font)
        
        # Display the timestamp (top-right).
        draw.text((width - 100, 10), f"{frame.timestamp:.1f}s", fill="black", font=font)
        
        images.append(np.array(img))
    
    # Save the sequence of images as a GIF.
    imageio.mimsave(filename, images, fps=fps)
    print(f"Saved GIF: {filename}")

# === Main Function Combining Generation, Tokenization, and Visualization ===
def main():
    # Set up canvas dimensions and number of frames.
    width, height = 640, 480
    num_frames = 50  # Adjust as desired.
    
    # 1. Generate a random trajectory with random target points and fuzz.
    original_frames = generate_random_trajectory(num_frames, width, height, segment_length=10)
    
    # 2. Encode the trajectory using our action encoder.
    encoded_actions = encode_for_tokenizer(original_frames, width, height)
    # Add a batch dimension for the tokenizer: shape becomes (1, num_frames, action_dim)
    action_data = encoded_actions[None, :, :]
    
    # 3. Load the Universal Action Tokenizer and tokenize.
    tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True)
    tokens = tokenizer(action_data)  # Tokenizes the batch of action chunks.
    
    # 4. Decode the tokens back into action data.
    decoded_actions = tokenizer.decode(tokens)  # Expected shape: (1, num_frames, action_dim)
    
    # Convert the decoded action matrix into a list of InteractionFrame objects.
    decoded_actions = decoded_actions[0]  # Remove the batch dimension.
    decoded_frames = []
    for i, action in enumerate(decoded_actions):
        frame = decode_action_vector(action, width, height)
        frame.timestamp = i / num_frames  # Reassign timestamp.
        decoded_frames.append(frame)
    
    # 5. Save GIFs for “before tokenization” and “after tokenization.”
    create_gif_from_frames(original_frames, "trajectory_before_tokenization.gif", width, height, fps=10)
    create_gif_from_frames(decoded_frames, "trajectory_after_tokenization.gif", width, height, fps=10)
    
    # 6. (Optional) Print token information.
    print("Token dim:", len(tokens[0]))
    print("Decoded actions shape:", decoded_actions.shape)

if __name__ == "__main__":
    main()
