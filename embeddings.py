import os

os.environ["HF_HOME"] = ".hf/hf_home"

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3", use_fp16=True
)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

sentences_1 = ["Hello"]

embedding = model.encode(
    sentences_1,
    batch_size=12,
    max_length=8192,  # If you don't need such a long length, you can set a smaller value to speed up the encoding process.
)["dense_vecs"]

print(embedding)
