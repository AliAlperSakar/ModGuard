from flask import Flask, request, jsonify
# Use a pipeline as a high-level helper
from transformers import pipeline
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM
import logging
# Use a pipeline as a high-level helper
from transformers import pipeline
import os
import openai
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import requests
import json

api_url = "https://vicuna-api.aieng.fim.uni-passau.de/v1/chat/completions"
api_model = "vicuna-33b-v1.3"
api_authorization = 'Bearer group6_ucdgu'

OPENAI_API_KEY = "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"
model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


#toxic_comment_
pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)



app = Flask(__name__)


@app.route('/moderate', methods=['POST'])
def moderate_text():
    # Extract the text from the POST request
    text = request.json['text']
    print("Came")
    # print(text)
    #Use both models (LLM and BERT) for moderation
    llm_result = moderate_with_llm(text)
    print(llm_result)
    bert_result = ""
    # bert_result = moderate_with_bert(text)

    # print('bert result:', bert_result[0]['label'], 'score: ', bert_result[0]['score'])
    

    # Compare the results and decide which one to use
    moderation_result = decide_moderation(llm_result, bert_result)
    logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    
    return jsonify({'moderation_result': moderation_result})



@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Implement code to gather and return relevant usage statistics
    metrics = gather_metrics()

    return jsonify(metrics)

def moderate_with_llm(text):  
    
    url = "https://vicuna-api.aieng.fim.uni-passau.de/v1/chat/completions"
    headers = {
    "Content-Type": "application/json",
    "Authorization": api_authorization,
    }

    # Set the data payload
    payload = json.dumps({
    "model": api_model,
    "temperature": 0.1,
    "language" : "english",
    "top_p":1,
    "frequency_penalty": 0.0,
    "max_tokens": 1500,
    "messages": [
        {"role": "system",
         "content": "You are tasked with the role of a content moderation system, responsible for analyzing user-submitted texts to identify the presence of undesirable or inappropriate language. Your task is to review each text and categorize it accordingly: if the text contains any form of toxic language, please respond with as not good. If the text is free of inappropriate content, respond with okay."
         },
         {"role": "user",
          "content": f"{text}"}
        ],    
    })
    # Send the POST request
    response = requests.post( url=url, headers=headers, data=payload)
    r = json.loads(response.text)
    # Print the response
    # print(response.json())
    print('LALAAAALAALAAA')
    print(r['choices'][0]['message']['content'])
    return response.json()
    # Use the LLM model to classify text

def moderate_with_bert(text):
    result = pipeline(text)
    return result

def decide_moderation(llm_result, bert_result):
    # Implement logic to decide which model's result to use
    # You can compare their confidence scores or use other criteria
    return 

def gather_metrics():
    # Implement code to gather usage statistics
    # Track requests, errors, processing times, etc.
    return 

if __name__ == '__main__':
    app.run(debug=True)
