import os
import gzip
import sys
import csv
import sys
from dotenv import load_dotenv

csv.field_size_limit(sys.maxsize)

os.environ["HF_HOME"] = ".hf/hf_home"

from FlagEmbedding import BGEM3FlagModel

load_dotenv()

core_path = os.getenv("CORE_PATH", "")
language = sys.argv[1]
core_file = sys.argv[2]
base_dir = os.path.dirname(os.path.abspath(__file__))
model_name = "BAAI/bge-m3"
model_file = model_name.replace("/", "_") + "_embeddings.tsv"

model = BGEM3FlagModel(model_name, use_fp16=True)


def embed(text):

    embedding = model.encode(
        [text],
        batch_size=12,
        max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
    )["dense_vecs"]

    embedding_str = " ".join([f"{x:.8f}" for x in embedding.flatten()])
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
