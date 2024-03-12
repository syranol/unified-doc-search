# Pipeline Intro
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I've been waiting for a HUggingFace course my whole life.")

print(res)

"""
Result:

No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision af0f99b
(https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
config.json: 100%|██████████████████████████████████████████████████████| 629/629 [00:00<00:00, 550kB/s]
model.safetensors: 100%|█████████████████████████████████████████████| 268M/268M [00:08<00:00, 31.2MB/s]
tokenizer_config.json: 100%|██████████████████████████████████████████| 48.0/48.0 [00:00<00:00, 121kB/s]
vocab.txt: 100%|█████████████████████████████████████████████████████| 232k/232k [00:00<00:00, 3.35MB/s]
[{'label': 'POSITIVE', 'score': 0.9598050713539124}]
"""

from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)

print(res)

'''
Truncation was not explicitly activated but `max_length` is provided a specific value, please use `truncation=True` to explicitly 
truncate examples to max length. Defaulting to 'longest_first' truncation strategy. If you encode pairs of sequences (GLUE-style)
with the tokenizer you can select this strategy more precisely by providing a specific strategy to `truncation`. Setting 
`pad_token_id` to `eos_token_id`:50256 for open-end generation. [{'generated_text': 'In this course, we will teach you how to start
working. To begin with, I am going to take you through three phases of work.\n'},
{'generated_text': 'In this course, we will teach you how to use Angular 2 with Angular 2 with Angular 2. If you want to use Angular 2 in your application'}]
'''

from transformers import pipeline

classifier = pipeline("zero-shot-classification")

res = classifier(
    "THis is a course about Python list comprehension",
    candidate_labels=["education", "politics", "business"],
)

print(res)
