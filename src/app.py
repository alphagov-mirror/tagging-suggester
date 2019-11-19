from flask import Flask, request, jsonify
from src.models.tagging_suggester import *

app = Flask(__name__)

tagging_suggester = TaggingSuggester()

@app.route('/create', methods=['POST'])
def tagging_suggestion():
    json = request.get_json()
    print(json)
    result = tagging_suggester.predict(json['text'])
    return jsonify({ "suggestions": result })

if __name__=="__main__":
    app.run(debug=True)
