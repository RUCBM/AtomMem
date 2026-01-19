from trl import SFTConfig,SFTTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, required=True)
parser.add_argument("--dataset_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--max_seq_length", type=int, default=4096)
parser.add_argument("--per_device_train_batch_size", type=int, default=1)
parser.add_argument("--save_steps", type=int, default=50)
parser.add_argument("--dataloader_drop_last", action='store_true')
parser.add_argument("--logging_steps", type=int, default=5)
parser.add_argument("--do_train", action='store_true')
parser.add_argument("--gradient_checkpointing_kwargs")
parser.add_argument("--fp16",action='store_true')

args = parser.parse_args()

dataset = load_dataset(args.dataset_path,split='train')
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
    )

training_args = SFTConfig(
    output_dir=args.output_dir,
    max_seq_length=args.max_seq_length,
    per_device_train_batch_size=args.per_device_train_batch_size,
    save_steps=args.save_steps,
    dataloader_drop_last=args.dataloader_drop_last,
    logging_steps=args.logging_steps,
    do_train=args.do_train,
    gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs,
    fp16=args.fp16
    )

trainer = SFTTrainer(
    model,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()