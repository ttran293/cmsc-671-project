from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import os
from dotenv import load_dotenv

print("="*60)
print("Starting LoRA Training Pipeline")
print("="*60)

load_dotenv()
hf_token = os.getenv('HUGGINGFACE_TOKEN')

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"\n[1/6] Loading dataset...")
dataset = load_dataset("json", data_files="summarized_dataset/train.jsonl")["train"]

# https://huggingface.co/docs/peft/en/developer_guides/lora

print(f"\n[2/6] Loading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
tokenizer.pad_token = tokenizer.eos_token

print(f"\n[3/6] Loading base model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    token=hf_token,
)

print(f"\n[4/6] Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj",
                    "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)

print(f"\n[5/6] Setting up training configuration...")
training_args = SFTConfig(
    output_dir="llama3-lora",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    max_seq_length=1024,
    logging_steps=1,
    save_steps=10,
)

def format_example(example):
    return f"User: {example['input']}\nAssistant: {example['output']}"

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    formatting_func=format_example,
)

print(f"\n[6/6] Starting training...")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print("="*60 + "\n")

trainer.train()

model.save_pretrained("llama3-lora")
tokenizer.save_pretrained("llama3-lora")
print("Model saved to: llama3-lora/")
