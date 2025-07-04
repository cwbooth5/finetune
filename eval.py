"""
put the model into eval mode for inference
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

peft_model = PeftModel.from_pretrained(base_model, "./lora-output")

peft_model.eval()

input_text = "### Question:\nWho wrote '1984'?\n### Answer:\n"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = peft_model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

