import os, json
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

BASE_MODEL = "google/flan-t5-large"  # small & cheap; swap if you have GPU budget

def format_example(ex):
    prompt = f"Instruction: {ex['instruction']}\nInput: {ex['input']}\nOutput:"
    return {"input_ids": tokenizer(prompt, truncation=True).input_ids,
            "labels": tokenizer(ex["output"], truncation=True).input_ids}

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", target_modules=["q","v","k","o"])
    model = get_peft_model(model, lora)

    ds = load_dataset("json", data_files="dataset.jsonl", split="train")
    ds = ds.map(format_example)

    args = TrainingArguments(
        output_dir="checkpoints/flan_t5_lora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        fp16=False,
        save_total_limit=2,
        logging_steps=20,
        report_to=[]
    )
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained("checkpoints/flan_t5_lora")
    tokenizer.save_pretrained("checkpoints/flan_t5_lora")
