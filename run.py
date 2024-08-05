import csv
import json
import re

import requests

import prompts

model_name = "situational-characteristics"

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


def generate_model():
    url = "http://localhost:11434/api/create"
    modelfile = f'''
        FROM llama3.1-70
        PARAMETER temperature 0.01
        SYSTEM """{prompts.SYSTEM}"""
    '''
    payload = {
        "name": model_name,
        "modelfile": modelfile,
    }
    response = requests.post(url, json=payload)

    print(response.text)

    if response.status_code == 200:
        return 1


def gen_sit_char(text):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompts.MESSAGE.format(text),
        "stream": False,
    }
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        output = response.json()["response"]
        result = parse_llm_output(output)

        return result


def process_tsv_file(input_file, output_file):
    with open(input_file, "r", newline="") as infile, open(
        output_file, "a", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")

        for row in reader:
            register, text = row
            sit_char = gen_sit_char(text[:5000])
            sit_char_str = dict_values_to_string(sit_char)
            print(register, sit_char_str)
            writer.writerow([register, sit_char_str])
            outfile.flush()  # Force write to disk


input_path = "/scratch/project_2010911/multilingual-CORE"
io_file = "fi/train.tsv"
input_file = f"{input_path}/{io_file}"

if generate_model():
    process_tsv_file(input_file, io_file)
else:
    print("Model generation failed")
