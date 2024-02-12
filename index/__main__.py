import glob

import fire
import torch
from llama_index import ServiceContext, VectorStoreIndex
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from llama_index.readers import PDFReader
from transformers import BitsAndBytesConfig


def main(hf_token: str, path: list, prompt: str):
    docs = []
    for p in glob.glob(path):
        docs.extend(PDFReader().load_data(p))

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    llm = HuggingFaceLLM(
        model_name="meta-llama/Llama-2-7b-chat-hf",
        tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
        query_wrapper_prompt=PromptTemplate("<s> [INST] {query_str} [/INST] "),
        context_window=3900,
        model_kwargs={"token": hf_token, "quantization_config": quantization_config},
        tokenizer_kwargs={"token": hf_token},
        device_map="auto",
    )

    service_context = ServiceContext.from_defaults(llm=llm, embed_model="local:BAAI/bge-small-en-v1.5")
    vector_index = VectorStoreIndex.from_documents(docs, service_context=service_context)
    query_engine = vector_index.as_query_engine(response_mode="compact")
    response = query_engine.query(prompt)
    return response


if __name__ == "__main__":
    fire.Fire(main)
