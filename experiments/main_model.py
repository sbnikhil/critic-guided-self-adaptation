from transformers import AutoModelForCausalLM, AutoTokenizer
 import loralib as lora

#from data_loader import load_tydiqa_by_language, get_available_languages

model_name = "Qwen/Qwen2.5-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto"
)

print(list(model.children()))