# fine tuning hijinks

This demonstrates a short workflow for training a very small LM locally. We use LoRA to train the model, and then we merge it with the original model. The questions and responses are just toy data, only used to verify our training worked as intended.

# Training

This is going to take the questions/responses inside `sample_train.jsonl` and train a 1B parameter model on them.

Use the following command:

```
uv run train.py
```

When that finishes, it will drop all the model files into `./lora-output`. 

# Saving the merged model

We can take the output from the training step and use that to save a new merged model.

Use this command:

```
uv run merge-model.py
```

When that's done, it's going to write out data to `./merged-model`.

