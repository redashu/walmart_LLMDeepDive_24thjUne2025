# using BPE subword tokenizer in openAI 
from transformers import GPT2Tokenizer
tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
myinput = "this is walmart LLM training session"

# Tokenize 
token=tokenizer.tokenize(myinput)
print(token)