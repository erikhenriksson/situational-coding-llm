import csv
import os
import re

import requests
import gzip
import time

import prompts

import subprocess

##### 1. Start the ollama server #####

# Set up ollama environment variables
env = os.environ.copy()
env["OLLAMA_TMPDIR"] = "/scratch/project_2010911/ollama"
env["OLLAMA_MODELS"] = "/scratch/project_2010911/ollama"

# Command to start the ollama server
command = "/users/ehenriks/bin/ollama serve"

# Split the command
cmd = command.split()

# Open a temp file to collect output
with open("ollama.txt", "w") as outfile:
    # Start the subprocess in the background, redirecting stdout and stderr
    proc = subprocess.Popen(cmd, stdout=outfile, stderr=subprocess.STDOUT, env=env)


##### 2. Define the model and parameters #####

model_name = "situationalcharacteristics"

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

# Function to convert dictionary values to a space-separated string
def dict_values_to_string(d):
    return " ".join(str(value) for value in d.values())


def parse_llm_output(text):
    # Process the output here!

    return text


def make_model():
    try:
        url = "http://localhost:11434/api/create"
        modelfile = f'''
                FROM llama3.1:70b
                PARAMETER temperature 0.01
                SYSTEM """{prompts.SYSTEM}"""
            '''
        payload = {
            "name": model_name,
            "modelfile": modelfile,
        }
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            return 1
        else:
            return 0
    except:
        return 0


def gen_sit_char(text):
    max_tries = 20
    tries = 0
    print("payload: ", prompts.MESSAGE.format(text))
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompts.MESSAGE.format(text),
        "stream": False,
    }
    while tries < max_tries:
        print("trying to generate")
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            output = response.json()["response"]
            result = parse_llm_output(output)
            return result
        else:
            print("unsuccesful", response.text)
        tries += 1
        time.sleep(5)
    else:
        print("Failed to generate text after 20 attempts.")
        exit()


def process_tsv_file(input_file, output_file):
    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file.replace(".gz", ""), "w", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            register, text = row
            # print processing register and first 10 chars of text
            print(register, text[:10])
            sit_char = gen_sit_char(text[:5000])
            sit_char_str = dict_values_to_string(sit_char)
            print(register, sit_char_str)
            writer.writerow([register, sit_char_str])
            outfile.flush()  # Force write to disk


input_path = "/scratch/project_2010911/multilingual-CORE"
io_file = "fi/train.tsv.gz"
input_file = f"{input_path}/{io_file}"


# Loop until generate_model returns 1 or tries exceed 20
max_tries = 20
tries = 0

print("waiting 5 seconds just to be sure")
time.sleep(5)
print("done")


while tries < max_tries:
    print("Trying to generate model")
    if generate_model() == 1:
        print("Model generation completed successfully.")
        break
    tries += 1
    time.sleep(5)
else:
    print("Failed to generate model after 20 attempts.")
    exit()

print("starting to process")
process_tsv_file(input_file, io_file)
