# Tokenization intro - Actually using tokenization
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

# 1 Default
res = classifier("I've been waiting for a HUggingFace course my whole life.")

print("Result from 1", res)

# 2 Model (But same)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

res = classifier("I've been waiting for a HUggingFace course my whole life.")

print("Result from 2", res)

# 3 Tokenize string or list of strings into mathematical representation the model understands
# [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102] -> 101 represents start and 102 is end
sequence = "Using a Transformer network is simple"
res = tokenizer(sequence)
print("Result from 3", res)
# 4 Token Spec - Return a list - Split string into words and their modifications
tokens = tokenizer.tokenize(sequence)
print("Result from 4", tokens)
# 5 Return Id - Return ID for each word
ids = tokenizer.convert_tokens_to_ids(tokens)
print("Result from 5", ids)
# 6 Return original strings - Return original string using ids
decoded_string = tokenizer.decode(ids)
print("Result from 6", decoded_string)

'''
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
Result from 1 [{'label': 'POSITIVE', 'score': 0.9598050713539124}]
Result from 2 [{'label': 'POSITIVE', 'score': 0.9598050713539124}]
Result from 3 {'input_ids': [101, 2478, 1037, 10938, 2121, 2897, 2003, 3722, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
Result from 4 ['using', 'a', 'transform', '##er', 'network', 'is', 'simple']
Result from 5 [2478, 1037, 10938, 2121, 2897, 2003, 3722]
Result from 6 using a transformer network is simple
'''