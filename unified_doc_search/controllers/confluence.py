import requests
import json
import logging
import os

from controllers.utils.data_sanitizer import sanitize_confluence

# Load environment variables from .env file
with open('.env') as f:
    for line in f:
        key, value = line.strip().split('=', 1)
        os.environ[key] = value

confluence_url = "https://shon4081.atlassian.net/wiki"
endpoint_url = f"{confluence_url}/rest/api/search"
access_token = os.environ.get("CONFLUENCE_TOKEN")
username = os.environ.get("CONFLUENCE_USERNAME")

def confluence_controller(query):
    #query = "capybara"
    # Define search parameters
    search_params = {
        "type": "page",
        "cql": f'text ~ "{query}" OR text ~ "{query}*"',
        "limit": 10
    }

    # Make the GET request with basic authentication
    response = requests.get(endpoint_url, params=search_params, auth=(username, access_token))

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON content from the response
        json_response = response.json()
        sanitized_data = sanitize_confluence(json_response)

        return sanitized_data, None
    else:
        # Print an error message if the request was not successful
        return None, response.text
