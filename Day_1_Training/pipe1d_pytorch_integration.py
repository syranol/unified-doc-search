# Tokenization intro - Actually using tokenization
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# PyTorch import
import torch
import torch.nn.functional as F

# Usually apply tokenizer directly instead of using separate functions
# This separation of steps is useful if you want to fine tune model with a 
# pytorch training loop.

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis")

X_train = ["I've been waiting for a HuggingFace course my whole life.",
           "Python is great!"]

res = classifier(X_train)
print("1", res)

batch = tokenizer(X_train, padding=True, truncation=True, max_length=512, return_tensors="pt")
# Dictionary of input_ids of tensor due to return_tensors specified, otherwise this would be a 
# normal list. But then we would have to take care of putting in the current format ourselves.
print("2", batch)

# Inference
with torch.no_grad():
    # Call model with unpacked batch as this is a dictionary 
    outputs = model(**batch)
    print("3", outputs)
    # Get prediction
    predictions = F.softmax(outputs.logits, dim=1)
    print("4", predictions)
    # Get label
    labels = torch.argmax(predictions, dim=1)
    print("5", labels)
    
'''
No model was supplied, defaulted to distilbert-base-uncased-finetuned-sst-2-english and revision
af0f99b (https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).
Using a pipeline without specifying a model name and revision in production is not recommended.
1 [{'label': 'POSITIVE', 'score': 0.9598050713539124}, {'label': 'POSITIVE', 'score': 0.9998615980148315}]
2 {'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
                           2607,  2026,  2878,  2166,  1012,   102],
                        [  101, 18750,  2003,  2307,   999,   102,     0,     0,     0,     0,     0,     0,
                           0,     0,     0,     0]]),
                        'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                                  [ , 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
3 SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
        [-4.2745,  4.6111]]), hidden_states=None, attentions=None)
4 tensor([[4.0195e-02, 9.5981e-01],
        [1.3835e-04, 9.9986e-01]])
5 tensor([1, 1])

'''