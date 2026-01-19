# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys

import datasets
from sympy import li
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import swanlab

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
    DataCollatorForCompletionOnlyLM
)

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()

    set_seed(training_args.seed)
    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_config}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    training_args.completion_only_loss = True
    training_args.report_to="swanlab"
    swanlab.login(api_key='WYylWBY5ChJPTM8zOmBMD')
    ################
    # Dataset
    ################
    dataset = load_dataset("json", data_files=script_args.dataset_name)

    # For ChatML we need to add special tokens and resize the embedding layer
    if (
        "<|im_start|>" in tokenizer.chat_template
        and "gemma-tokenizer-chatml" not in tokenizer.name_or_path
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
        model_kwargs = None

    # def formatting_prompts_func(examples):
    #     output_texts = []
    #     for i in range(len(examples["prompt"])):
    #         output_texts.append(examples["prompt"][i]+ '<|vision_pad|>' + examples["completion"][i])
    #     return output_texts
    # response_template = [151654]

    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_config)
    )
    # formatting_func=formatting_prompts_func,
    #     data_collator=collator,
    # torch.cuda.empty_cache()
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)
