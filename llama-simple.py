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

model_file = "llama-3.1-8b.tsv"

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

batch_size = 4
max_text_len = 5000


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
    explanations_list = []
    for model_output in batch_outputs:
        # Extract scores from the text
        scores = []
        explanations = []
        for line in model_output.strip().split("\n"):
            if line.strip().startswith("["):
                try:
                    score = str(int(line.split("[")[2].split("]")[0]))
                    scores.append(score)
                    explanation = line.split("(")[1].split(")")[0]
                    explanations.append(explanation)
                except:
                    print("Error parsing this line:", line)
                    exit()

        scores_list.append(" ".join(scores))
        explanations_list.append("|".join(explanations))

    return scores_list, explanations_list


def process_tsv_file(input_file, output_file):
    def write_batch_results(batch, scores, explanations, writer):
        for i, row in enumerate(batch):
            register, text = row
            score = scores[i]
            explanation = explanations[i]
            print(register, score, explanation, text[:50])
            writer.writerow([register, score, explanation, text])

    proc_rows = 0
    last_row = []
    with open(output_file, "r", newline="") as outfile_temp:
        reader_temp = [row for row in csv.reader(outfile_temp, delimiter="\t") if row]
        proc_rows = len(reader_temp)

        last_row = reader_temp[-1][-1]

    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file, "a", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        batch = []
        row_i = 1
        active = 0
        for row in reader:

            if active:
                batch.append(row)
                if len(batch) == batch_size:
                    texts = [text[:max_text_len] for _, text in batch]
                    scores, explanations = generate_responses(texts)
                    write_batch_results(batch, scores, explanations, writer)
                    batch = []
                    outfile.flush()

            if row_i == proc_rows:
                if last_row != row[-1]:
                    print("Mismatching rows")
                    exit()
                else:
                    active = 1

            row_i += 1
        # Process any remaining rows in the last batch
        if batch:
            texts = [text[:max_text_len] for _, text in batch]
            scores, explanations = generate_responses(texts)
            write_batch_results(batch, scores, explanations, writer)

        outfile.flush()


input_path = "/scratch/project_2010911/multilingual-CORE"
io_file = "fi/train.tsv.gz"
input_file = f"{input_path}/{io_file}"
output_file = io_file.replace(".tsv.gz", "") + f"_{model_file}"


process_tsv_file(input_file, output_file)
