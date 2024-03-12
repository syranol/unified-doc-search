import sys 
import os
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

from flask import Flask, jsonify, request, render_template
from controllers.slack import slack_controller
from controllers.confluence import confluence_controller
from nlp.transformer import transform_result

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')
    
@app.route('/search', methods=['GET'])
def search():
    # Extract parameters from the request
    query = request.args.get('q')

    # Search Slack
    slack_data, slack_result_code = slack_search(query)

    # Search Confluence 
    confluence_data, confluence_result_code = confluence_search(query)

    combined_data = {}
    
    if slack_result_code == 200:
        combined_data.update(slack_data)
    
    if confluence_result_code == 200:
        combined_data.update(confluence_data)
    
    # if slack_result_code == 200 or confluence_result_code == 200:
    #     return jsonify(combined_data), 200
    # else:
    #     return jsonify({'error': 'Some error occurred'}), 500

    result = transform_result(query, combined_data)
    
    return result, 200
    #return jsonify({'result': f'{result}'}), 200


@app.route('/slack/search', methods=['GET'])
def slack_search(query):
    # Extract parameters from the request
    #query = request.args.get('q')

    # Perform the Slack search
    result, error = slack_controller(query)

    if result is not None:
        return result, 200
    else:
        return error, 500
        
@app.route('/confluence/search', methods=['GET'])
def confluence_search(query):
    # Extract parameters from the request
    #query = request.args.get('q')

    # Perform the Confluence search
    result, error = confluence_controller(query)

    if result is not None:
        return result, 200
    else:
        return error, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)