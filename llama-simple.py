import os
import prompts
import csv
import gzip
import re

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["XDG_CACHE_HOME"] = ".hf/xdg_cache_home"
os.environ["HF_DATASETS_CACHE"] = ".hf/datasets_cache"
os.environ["WANDB_DISABLED"] = "true"

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")

# Model ID from Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load tokenizer and model from Hugging Face Hub (requires access token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id, token=access_token, torch_dtype=torch.bfloat16, device_map="auto"
)

# Define conversation termination tokens
terminators = [
    tokenizer.eos_token_id,  # End-of-sentence token
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Custom end-of-conversation token
]

# Maximum allowed input token length
MAX_INPUT_TOKEN_LENGTH = 4096


def generate_response(context):
    prompt = [
        {
            "role": "system",
            "content": prompts.SYSTEM,
        },
        {"role": "user", "content": prompts.MESSAGE.format(context)},
    ]

    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    outputs = model.generate(
        **inputs, do_sample=False, temperature=0, max_new_tokens=512
    )
    output_parsed = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]

    return output_parsed


def process_tsv_file(input_file, output_file):
    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file.replace(".gz", ""), "w", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            register, text = row
            model_output = generate_response(text[:5000])
            print(model_output)
            exit()

            sit_char_str = dict_values_to_string(sit_char)
            print(register, sit_char_str)
            writer.writerow([register, sit_char_str])
            outfile.flush()  # Force write to disk


input_path = "/scratch/project_2010911/multilingual-CORE"
io_file = "fi/train.tsv.gz"
input_file = f"{input_path}/{io_file}"

process_tsv_file(input_file, io_file)
