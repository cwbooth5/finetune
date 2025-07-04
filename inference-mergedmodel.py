from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged-model")
tokenizer = AutoTokenizer.from_pretrained("./merged-model")

question = input()

input_text = "### Question:\nWhat is 2 + 2?\n### Answer:\n"
input_text = "### Question:\nHow do python context managers work?\n### Answer:\n"

input_text = f"### Question:\n{question}\n### Answer:\n"

inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
