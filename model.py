# model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# Use correct model version from HuggingFace
model_id = "teknium/OpenHermes-2.5-Mistral-7B"

# Optional: use 4-bit quantization for better performance on lower GPUs
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config
)

# Wrap as HuggingFace text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.2
)
