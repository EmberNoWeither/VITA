from flask import Flask, request, jsonify
import requests
import json
app = Flask(__name__)


"""
key_file.json
{
  "management": {
    "key": "TAVXKQF1",
    "expiration time": "2024-12-17T08:33:40.991793Z"
  },
  "inference": {
    "key": "v1pMDtKM",
    "expiration time": "2024-12-17T08:33:40.991775Z"
  },
  "API": {
    "key": "PULEzRbH"
  }
}
"""
# with open('key_file.json') as f:
#     keys = json.load(f)
#     management_key = keys['management']['key']
#     inference_key = keys['inference']['key']
#     api_key = keys['API']['key']



@app.route('/vita_qa', methods=['POST'])
def qa():
    data = request.json
    response = requests.post(
        url="http://localhost:8090/predictions/vita",
        headers={'Content-Type': 'application/json'},
        json=data
    )

    print("\nContent:")
    print(response.text)
    result = {
        'output':response.text
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)