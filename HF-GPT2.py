import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Rave271/f1-gpt2-finetuned"

# Load model + tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GPT-2 pad fix
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 🔥 Reduced copy-bias prompt (no repeated ranking blocks)
DEFAULT_PROMPT = """F1 2024 Constructor Championship Prediction

Recent Champions:
2021 Champion: Mercedes (613pts)
2022 Champion: Red Bull (759pts)
2023 Champion: Red Bull (860pts)

Current Standings After Round 15:
1. Red Bull (500pts)
2. Mercedes (320pts)
3. Ferrari (300pts)

Final Result:
"""

def generate(prompt):
    model.eval()

    # Anchor structure
    prompt = prompt.rstrip()
    if not prompt.endswith("1."):
        prompt = prompt + "\n1."

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.25,              # Lower = more stable
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,        # Prevent copying
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract only ranking lines
    lines = full_text.split("\n")
    result_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith(tuple(str(i) + "." for i in range(1, 11))):
            result_lines.append(line)
        if len(result_lines) == 10:
            break

    return "\n".join(result_lines)


interface = gr.Interface(
    fn=generate,
    inputs=gr.Textbox(value=DEFAULT_PROMPT, lines=20, label="Input Prompt"),
    outputs="text",
    title="F1 GPT-2 Championship Predictor",
)

interface.launch()