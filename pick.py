import dill
from transformers import pipeline
import torch

# Create the pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Pickle the pipeline object
with open("pipeline.pkl", "wb") as f:
    dill.dump(pipe, f)

print('done')
with open("pipeline.pkl", "rb") as f:
    loaded = cloudpickle.load(f)

messages = [
    {
        "role": "system",
        "content": "You are a data generator. Generate accurate data according to the query. ",
    },
    {"role": "user", "content": "Generate example data in json format for a inteeligent warehouse system."},
]
prompt = loaded.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
outputs = loaded(
    prompt, max_new_tokens=555, do_sample=True, temperature=0.7, top_k=50, top_p=0.95
)
print(outputs[0]["generated_text"])