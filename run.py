from datasets import load_dataset
from transformers import Trainer, AutoModelForCausalLM, GPT2Config, TrainingArguments

dataset = load_dataset('wikipedia')

model_config = GPT2Config(n_layer=6, n_head=6, n_positions=1024, n_embd=384)
model = AutoModelForCausalLM.from_config(model_config)

training_args = TrainingArguments(
    output_dir='./out',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
)
