"""
merge the output weights into the model and save the model
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# Load base model + tokenizer
base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-rw-1b")

# Load LoRA adapter
peft_model = PeftModel.from_pretrained(base_model, "./lora-output")

peft_model.eval()

# at this point we could run inference using our new weights
# do it just to show that the model is trained using our data.
input_text = "### Question:\nWho wrote '1984'?\n### Answer:\n"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = peft_model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Load again
base_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-rw-1b")
model = PeftModel.from_pretrained(base_model, "./lora-output")

# Merge LoRA into base
merged_model = model.merge_and_unload()

merged_model.save_pretrained("./merged-model")
tokenizer.save_pretrained("./merged-model")

# Now we have the merged model on disk and pick that as an LM for whatever we like.
