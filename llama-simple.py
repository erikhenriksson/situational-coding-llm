import os
import prompts
import csv
import gzip
import re

os.environ["HF_HOME"] = ".hf/hf_home"
os.environ["WANDB_DISABLED"] = "true"

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

load_dotenv()

access_token = os.getenv("HUGGINGFACE_ACCESS_TOKEN", "")

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=access_token,
    # torch_dtype=torch.bfloat16,
    device_map="auto",
    # quantization_config=quantization_config,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def generate_responses(contexts):
    messages = [
        [
            {"role": "system", "content": prompts.SYSTEM},
            {"role": "user", "content": prompts.MESSAGE.format(context)},
        ]
        for context in contexts
    ]

    # Tokenize the batch of prompts with padding and truncation
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        attn_implementation="flash_attention_2",
        padding=True,  # Enable padding
        truncation=True,  # Enable truncation
    ).to("cuda")

    # Generate responses for the batch
    outputs = model.generate(
        **inputs, do_sample=True, temperature=0.01, max_new_tokens=1024
    )

    # Decode the batch of outputs
    batch_outputs = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    scores_list = []
    for model_output in batch_outputs:
        # Extract scores from the text
        scores = []

        for line in model_output.strip().split("\n"):
            try:
                score = str(int(line.split("[")[2].split("]")[0]))
                scores.append(score)
            except:
                print("Error parsing this line:", line)
                exit()

        scores_list.append(" ".join(scores))

    return batch_outputs


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
        attn_implementation="flash_attention_2",
    ).to("cuda")

    outputs = model.generate(
        **inputs, do_sample=True, temperature=0.01, max_new_tokens=1024
    )
    model_output = tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]


"""

def process_tsv_file(input_file, output_file):
    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file.replace(".gz", ""), "w", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            register, text = row
            model_output, scores = generate_response(text[:5000])

            # ... code continues

            print(model_output, scores)
            exit()

            sit_char_str = dict_values_to_string(sit_char)
            print(register, sit_char_str)
            writer.writerow([register, sit_char_str])
            outfile.flush()  # Force write to disk

"""

batch_size = 4
max_text_len = 5000


def process_tsv_file(input_file, output_file):
    def write_batch_results(batch, batch_outputs, writer):
        for i, row in enumerate(batch):
            register, text = row
            model_output = batch_outputs[i]
            writer.writerow([register, model_output, text])

    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file.replace(".gz", ""), "w", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == batch_size:
                texts = [text[:max_text_len] for _, text in batch]
                batch_outputs = generate_responses(texts)
                write_batch_results(batch, batch_outputs, writer)
                batch = []
        outfile.flush()
        # Process any remaining rows in the last batch
        if batch:
            texts = [text[:max_text_len] for _, text in batch]
            batch_outputs = generate_responses(texts)
            write_batch_results(batch, batch_outputs, writer)

        outfile.flush()


input_path = "/scratch/project_2010911/multilingual-CORE"
io_file = "fi/train.tsv.gz"
input_file = f"{input_path}/{io_file}"

process_tsv_file(input_file, io_file)
