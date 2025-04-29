from transformers import pipeline
import torch

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
pipeline = pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")

messages = [
    {"role": "system", "content": "You are a chatbot with expertise in summarizing documents"},
    {"role": "user", "content": "Provide a lay summary of this abstract: Metabolic reprogramming enables tumour cells to sustain their continuous proliferation and adapt to the ever-changing microenvironment. Branched-chain amino acids (BCAAs) and their metabolites are involved in intracellular protein synthesis and catabolism, signal transduction, epigenetic modifications, and the maintenance of oxidative homeostasis. Alterations in BCAA metabolism can influence the progression of various tumours. However, how BCAA metabolism is dysregulated differs among depending on tumour type; for example, it can manifest as decreased BCAA metabolism leading to BCAA accumulation, or as enhanced BCAA uptake and increased catabolism. In this review, we describe the role of BCAA metabolism in the progression of different tumours. As well as discuss how BCAA metabolic reprogramming drives tumour therapy resistance and evasion of the antitumour immune response, and how these pro-cancer effects are achieved in part by activating the mTORC signalling pathway. In-depth investigations into the potential mechanisms by which BCAA metabolic reprogramming affects tumorigenesis and tumour progression can enhance our understanding of the relationship between metabolism and cancer and provide new strategies for cancer therapy."},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1])

