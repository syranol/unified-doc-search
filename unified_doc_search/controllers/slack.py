import json
import logging
import os
from pathlib import Path

import requests

from .utils.data_sanitizer import sanitize_slack

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def _load_env():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    with env_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)


_load_env()

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
