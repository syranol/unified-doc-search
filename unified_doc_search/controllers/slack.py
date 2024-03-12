import requests
import logging
import json
import pprint
import os

from controllers.utils.data_sanitizer import sanitize_slack

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
with open('.env') as f:
    for line in f:
        key, value = line.strip().split('=', 1)
        os.environ[key] = value

slack_token = os.environ.get("SLACK_ACCESS_TOKEN")

def slack_controller(query):
    base_url = "https://syranol.slack.com/api/search.all"
    headers = {
        "Authorization": f"Bearer {slack_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "query": f'{query}*',
        "sort": "score"
    }
    response = requests.post(base_url, headers=headers, data=data)

    if response.status_code == 200:
        
        json_response = response.json()
        sanitized_data = sanitize_slack(json_response)

        return sanitized_data, None
    else:
        return None, response.text
