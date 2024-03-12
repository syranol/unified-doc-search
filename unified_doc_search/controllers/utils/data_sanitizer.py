from urllib.parse import urljoin

base_confluence_url = "https://shon4081.atlassian.net/wiki"

import logging

def sanitize_confluence(data):
    
    desired_data = {}
    
    for idx, result in enumerate(data["results"]):
        
        # Store link to documentation
        result_link =  result["content"]["_links"]["webui"]
        full_link = urljoin(base_confluence_url, result_link)
        
        # Store text that matches search
        text = result["excerpt"]
        
        desired_data[f'Confluence_{idx}'] = {
            "link": full_link,
            "text": text,
            "source": "Confluence"
        }
    
    return desired_data
                     
def sanitize_slack(data):
    desired_data = {}

    for idx, result in enumerate(data["messages"]["matches"]):
        # Store link to documentation
        result_link =  result["permalink"]
        
        # Store text that matches search
        text = result["blocks"][0]["elements"][0]["elements"][0]["text"]
        
        desired_data[f'Slack_{idx}'] = {
            "link": result_link,
            "text": text,
            "source": "Slack"
        }

    return desired_data
    