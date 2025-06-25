# !pip install openai tiktoken
from openai import OpenAI
import json
import tiktoken # for token counting
import numpy as np
from collections import defaultdict

# 
# using openai subscription api_key
ashu_apiKey = ""
# Initializing openai with key
ashu_api_client = OpenAI(api_key=ashu_apiKey)

data_path =  "/content/ashudataset.jsonl"

# Load the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

# Initial dataset stats
print("Num examples:", len(dataset))
print("First example:")
for message in dataset[0]["messages"]:
    print(message)
## checking data format 

# Format error checks
format_errors = defaultdict(int)

for ex in dataset:
    if not isinstance(ex, dict):
        format_errors["data_type"] += 1
        continue

    messages = ex.get("messages", None)
    if not messages:
        format_errors["missing_messages_list"] += 1
        continue

    for message in messages:
        if "role" not in message or "content" not in message:
            format_errors["message_missing_key"] += 1

        if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
            format_errors["message_unrecognized_key"] += 1

        if message.get("role", None) not in ("system", "user", "assistant", "function"):
            format_errors["unrecognized_role"] += 1

        content = message.get("content", None)
        function_call = message.get("function_call", None)

        if (not content and not function_call) or not isinstance(content, str):
            format_errors["missing_content"] += 1

    if not any(message.get("role", None) == "assistant" for message in messages):
        format_errors["example_missing_assistant_message"] += 1

if format_errors:
    print("Found errors:")
    for k, v in format_errors.items():
        print(f"{k}: {v}")
else:
    print("No errors found")

# dataset path
dataset_path = "/content/ashudataset.jsonl"
# fine tuning dataset_path using gpt4o-mini
response = ashu_api_client.files.create(
    file=open(dataset_path, "rb"),
    purpose='fine-tune'
)
# printing file id
print(response.id)

# list events of fine tuning progress
response = ashu_api_client.fine_tuning.jobs.list_events(fine_tuning_job_id=response.id)
# printing events
print(response)

model_suffix_name = "codebased-ashu-sarcasticmodel"
# creating fine tuning job
response = ashu_api_client.fine_tuning.jobs.create(
    training_file=response.id,
    model="gpt-4o-mini-2024-07-18",
    suffix=model_suffix_name,
    hyperparameters={
        "n_epochs": 5,

    }
)
# printing model id
print(response)
print(response.id)

# list fine tuned model name
mymodels = ashu_api_client.models.list()
print(mymodels)
for model in mymodels.data:
  if "ashu" in model.id:
    print(model.id)