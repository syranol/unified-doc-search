import torch
import numpy as np

from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict

# Load pre-trained RoBERTa model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)

# Load the saved state dict
model.load_state_dict(torch.load('trained_model.pth'))
# eval() for official use
model.eval()

def transform_result(query, combined_data):

    ranking = rank_result(query, combined_data)
    final_result = sort_result(combined_data, ranking)

    return final_result
    
def sort_result(combined_data, ranking):
    sorted_result = []

    for key, score in ranking:
        if key in combined_data:
            updated_data = combined_data[key].copy()
            updated_data["score"] = float(score)
            sorted_result.append(updated_data)
    print(sort_result)
    return sorted_result
    
def rank_result(query, combined_data):

    # Query
    
    # Tokenization
    query_tokens = tokenizer(query, return_tensors="pt")
    # Vector representation of the query 
    query_embedding = model(**query_tokens).last_hidden_state.mean(dim=1).detach().cpu().numpy()


    # Combined Data
    
    combined_data_embeddings = []
    # Search by 'text' from combined_data of Slack and Confluence
    for data_key, data_value in combined_data.items():
        # Tokenization
        combined_data_tokens = tokenizer(data_value['text'], return_tensors="pt")
        # Pass the tokenized input to model and extracts
        result_embedding = model(**combined_data_tokens).last_hidden_state.mean(dim=1).detach().cpu().numpy()
        combined_data_embeddings.append(result_embedding.squeeze())

    # Convert the NumPy array to PyTorch tensor
    # This is for data type consistency 
    query_embedding = torch.tensor(query_embedding.squeeze()) 
    combined_data_embeddings = torch.tensor(np.stack(combined_data_embeddings).squeeze()) 
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], combined_data_embeddings).flatten()

    # Rank search results based on similarity scores
    ranked_results = sorted(zip(combined_data, similarities), key=lambda x: x[1], reverse=True)

    return ranked_results

''' 
Example input

query = "Capybara"
search_results = [
                    "Capybara",
                    "Capybara are animals",
                    "Capybara are from South America",
                    "animals known as Capybara",
                    "oranges",
                    "dogs"
                ]
                
Example output

[('Capybara', 0.9999999),
 ('Capybara are animals', 0.97575915),
 ('oranges', 0.971982),
 ('animals known as Capybara', 0.959948),
 ('Capybara are from South America', 0.9588328),
 ('dogs', 0.9573278)]
 '''