import os
import gzip
import sys
import csv
import sys
from dotenv import load_dotenv

csv.field_size_limit(sys.maxsize)

os.environ["HF_HOME"] = ".hf/hf_home"


import torch
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig


load_dotenv()

core_path = os.getenv("CORE_PATH", "")
language = sys.argv[1]
core_file = sys.argv[2]
base_dir = os.path.dirname(os.path.abspath(__file__))
model_name = "BAAI/bge-multilingual-gemma2"
model_file = model_name.replace("/", "_") + "_embeddings.tsv"
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModel.from_pretrained(model_name, quantization_config=quantization_config)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def embed(text):

    max_length = 8192

    batch_dict = tokenizer(
        [text],
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**batch_dict)
        embedding = last_token_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        embedding_str = " ".join([f"{x:.16f}" for x in embedding.flatten()])
        return embedding_str


def process_tsv_file(input_file):
    output_file = f"{base_dir}/data/{language}/{model_file}"
    proc_rows = 0
    last_row = []
    try:
        with open(output_file, "r", newline="") as outfile_temp:
            reader_temp = [
                row for row in csv.reader(outfile_temp, delimiter="\t") if row
            ]
            proc_rows = len(reader_temp)

            last_row = reader_temp[-1][-1]
    except:
        pass
    print(proc_rows)
    with gzip.open(input_file, "rt", encoding="utf-8") as infile, open(
        output_file, "a", newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter="\t")
        writer = csv.writer(outfile, delimiter="\t")
        row_i = 1
        active = 0
        if not proc_rows:
            active = 1
        for row in reader:
            if active:
                register, text = row
                embedding = embed(text[:5000])
                writer.writerow([register, embedding, text])
                outfile.flush()  # Force write to disk

            if row_i == proc_rows:
                if last_row != row[-1]:
                    print("Mismatching rows")
                    exit()
                else:
                    active = 1

            row_i += 1


input_file = f"{core_path}/{language}/{core_file}"

process_tsv_file(input_file)
