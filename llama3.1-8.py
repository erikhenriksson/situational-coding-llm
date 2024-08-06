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
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig, pipeline

load_dotenv()

login(token=os.getenv("HUGGINGFACE_ACCESS_TOKEN", ""))

"""
model_id = "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=10000,  # Note: Update this as per your use-case
    do_fuse=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    quantization_config=quantization_config,
)
"""

"""
# Load the model and tokenizer
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Move the model to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

"""

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.bfloat16,
    },
    device_map="auto",
)


sit_char_params = {
    "a spoken transcript": 0,
    "lyrical or artistic": 0,
    "pre-planned and edited": 0,
    "interactive": 0,
    "is an expert": 0,
    "focuses on himself/herself": 0,
    "assumes technical background knowledge": 0,
    "assumes cultural/social knowledge": 0,
    "assumes personal knowledge about himself/herself": 0,
    "narrate past events": 0,
    "explain information": 0,
    "describe a person, place, thing or idea": 0,
    "persuade the reader": 0,
    "entertain the reader": 0,
    "sell a product or service": 0,
    "give advice or recommendations": 0,
    "provide 'how-to' instructions": 0,
    "express opinions": 0,
    "common knowledge": 0,
    "direct quotes": 0,
    "factual/scientific evidence": 0,
    "opinion": 0,
    "personal experience": 0,
}


def dict_values_to_string(d):
    return " ".join(str(value) for value in d.values())


def parse_llm_output(text):
    # Split the text into lines
    lines = text.strip().split("\n")

    # Regular expression pattern to match the format
    pattern = r"- (.*?) \[(.*?)\] \((.*?)\)"

    # Dictionary to store the results
    result = {key: 0 for key in sit_char_params.keys()}

    for line in lines:
        match = re.match(pattern, line)
        if match:
            parameter, score, explanation = match.groups()
            # Remove the leading hyphen and space from the parameter name
            parameter = parameter.lstrip("- ")

            # Ensure there is no colon
            parameter = parameter.split(":")[-1].strip()

            if parameter in result:
                # result[parameter] = {"score": int(score), "explanation": explanation}
                result[parameter] = int(score)
            else:
                print('Error: "{}" not found in the parameters'.format(parameter))
                print("Full response: ", text)
                exit()

    if any(value == 0 for value in result.values()):
        print()
        print("Warn: Some parameters are missing")
        print("Full response: ", text)
        print(result)

    return result


def generate(text):
    prompt = [
        {
            "role": "system",
            "content": prompts.SYSTEM,
        },
        {"role": "user", "content": prompts.MESSAGE.format(text)},
    ]

    outputs = pipe(
        prompt,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.01,
    )
    assistant_response = outputs[0]["generated_text"][-1]["content"]

    result = parse_llm_output(assistant_response)

    return result


def process_tsv_file(input_file, output_file):
    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file.replace(".gz", ""), "w", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            register, text = row
            sit_char = generate(text[:5000])
            sit_char_str = dict_values_to_string(sit_char)
            print(register, sit_char_str)
            writer.writerow([register, sit_char_str])
            outfile.flush()  # Force write to disk


input_path = "/scratch/project_2010911/multilingual-CORE"
io_file = "fi/train.tsv.gz"
input_file = f"{input_path}/{io_file}"

process_tsv_file(input_file, io_file)
