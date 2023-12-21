from flask import Flask, request, jsonify, Response
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
import sqlite3
import time
from flask_swagger_ui import get_swaggerui_blueprint
from prometheus_client import Counter, start_http_server


conn = sqlite3.connect('response_times.db', timeout=20)

api_url = "https://vicuna-api.aieng.fim.uni-passau.de/v1/chat/completions"
api_model = "vicuna-33b-v1.3"
api_authorization = 'Bearer group6_ucdgu'

OPENAI_API_KEY = "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"
model_path = "martin-ha/toxic-comment-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Initialize the SQLite database and create a table if it doesn't exist
conn = sqlite3.connect('response_times.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS response_times (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        response_time REAL
    )
''')
conn.commit()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS error_rates (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        error_rate INTEGER
    )
''')
conn.commit()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS request_volumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS moderation_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        decision TEXT
    )
''')
conn.commit()


cursor.execute('''
    CREATE TABLE IF NOT EXISTS prompt_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        prompt_tokens INTEGER
    )
''')
conn.commit()


conn.close()


#toxic_comment_
pipeline =  TextClassificationPipeline(model=model, tokenizer=tokenizer)



logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

app = Flask(__name__)

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Our Swagger schema file (static/swagger.json)

# Call factory function to create our blueprint
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  # Swagger UI config overrides
        'app_name': "My Flask API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

error_counter = Counter('my_failures', 'Description of counter')  # Create a Prometheus Counter


@app.route('/moderate', methods=['POST'])
def moderate_text():
    # Extract the text from the POST request
    # Record the start time
    start_time = time.time()

    app.logger.info('Info level log')
    app.logger.warning('Warning level log')

    
    text = request.json['text']
    print("Came")
    # print(text)
    #Use both models (LLM and BERT) for moderation
    llm_result = moderate_with_llm(text)
    bert_result = ""
    # bert_result = moderate_with_bert(text)

    # print('bert result:', bert_result[0]['label'], 'score: ', bert_result[0]['score'])
    

    # Compare the results and decide which one to use
    moderation_result = decide_moderation(llm_result, bert_result)
    
    
    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    print("ALPER ->  ", moderation_result['choices'][0]['message']['content'])
    # Store the response time in the database
    conn = sqlite3.connect('response_times.db')
    cursor = conn.cursor()
    for _ in range(5):  # Retry up to 5 times
        try:
            cursor.execute('INSERT INTO response_times (response_time) VALUES (?)', (elapsed_time,))
            conn.commit()
            break
        except sqlite3.OperationalError as e:
            print(f"Database locked, retrying...: {e}")
            time.sleep(1)  # Wait for 1 second before retrying

    # Continue with other database operations
    cursor.execute('INSERT INTO request_volumes DEFAULT VALUES')
    cursor.execute('INSERT INTO moderation_decisions (decision) VALUES (?)', (moderation_result['choices'][0]['message']['content'],))

    conn.commit()
    conn.close()
    # Return the response along with the response time
    response_data = {'moderation_result': moderation_result, 'response_time_seconds': elapsed_time}
    return jsonify(response_data)




@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Retrieve all response times from the database
    conn = sqlite3.connect('response_times.db')
    cursor = conn.cursor()
    
    # Retrieve and calculate response time metrics
    cursor.execute('SELECT response_time FROM response_times')
    response_time_rows = cursor.fetchall()
    response_time_sum = sum(row[0] for row in response_time_rows)
    response_time_count = len(response_time_rows)
    
    # Retrieve and calculate error rate metrics
    cursor.execute('SELECT error_rate FROM error_rates')
    error_rate_rows = cursor.fetchall()
    error_rate_sum = sum(row[0] for row in error_rate_rows)
    error_rate_count = len(error_rate_rows)
    cursor.execute('SELECT decision, COUNT(*) FROM moderation_decisions GROUP BY decision')
    moderation_decision_counts = cursor.fetchall()

    cursor.execute('SELECT COUNT(*) FROM request_volumes')
    request_volume = cursor.fetchone()[0]

    # Create the Prometheus metrics exposition format
    metrics = [
        '# HELP acme_http_router_request_seconds Latency though all of ACME\'s HTTP request router.',
        '# TYPE acme_http_router_request_seconds summary',
        f'acme_http_router_request_seconds_sum{{path="/api/v1",method="GET"}} {response_time_sum}',
        f'acme_http_router_request_seconds_count{{path="/api/v1",method="GET"}} {response_time_count}',
        '# HELP acme_error_rates Error rates observed in the system.',
        '# TYPE acme_error_rates summary',
        f'acme_error_rates_sum{{path="/api/v1",method="GET"}} {error_rate_sum}',
        f'acme_error_rates_count{{path="/api/v1",method="GET"}} {error_rate_count}',
    ]

    metrics.append(f'# HELP acme_request_volumes Total number of requests received.')
    metrics.append(f'# TYPE acme_request_volumes counter')
    metrics.append(f'acme_request_volumes_total{{path="/api/v1",method="POST"}} {request_volume}')


    for decision, count in moderation_decision_counts:
        metrics.append(f'# HELP acme_moderation_decisions_count Count of moderation decisions.')
        metrics.append(f'# TYPE acme_moderation_decisions_count counter')
        metrics.append(f'acme_moderation_decisions_count{{decision="{decision}"}} {count}')


    # Retrieve and calculate prompt tokens metrics
    cursor.execute('SELECT prompt_tokens FROM prompt_tokens')
    prompt_tokens_rows = cursor.fetchall()
    
    conn.close()

    prompt_tokens_sum = sum(row[0] for row in prompt_tokens_rows if row[0] is not None)  # Ensure non-None values
    prompt_tokens_count = len([row[0] for row in prompt_tokens_rows if row[0] is not None])  # Ensure non-None values

    # Create the Prometheus metrics exposition format
    metrics.append(f'# HELP acme_prompt_tokens Total number of prompt tokens used.')
    metrics.append(f'# TYPE acme_prompt_tokens counter')
    metrics.append(f'acme_prompt_tokens_total{{path="/api/v1",method="POST"}} {prompt_tokens_sum}')

    # Join the metrics into a single string with newline separators
    metrics_response = '\n'.join(metrics)
    
    # Return the metrics as plain text
    return Response(metrics_response, mimetype='text/plain')

def moderate_with_llm(text):  
    
    url = "https://vicuna-api.aieng.fim.uni-passau.de/v1/chat/completions"
    headers = {
    "Content-Type": "application/json",
    "Authorization": api_authorization,
    }

    # Set the data payload
    payload = json.dumps({
    "model": api_model,
    "temperature": 0,
    "language" : "english",
    "frequency_penalty": 0.0,
    # "max_tokens": 1500,
    "messages": [
        {"role": "system",
         "content": "You are tasked with the role of a content moderation system, responsible for analyzing user-submitted texts to identify the presence of undesirable or inappropriate language. Your task is to review each text and categorize it accordingly: if the text contains any form of toxic language, please respond with 'toxic'. If the text is free of inappropriate content, respond with 'non-toxic'. Answer only with lowercase and without punctuation."
         },
         {"role": "user",
          "content": text}
        ],    
    })
    conn = sqlite3.connect('response_times.db')
    cursor = conn.cursor()
    # Send the POST request
    response = requests.post( url=url, headers=headers, data=payload)
    if response.status_code != 200:
        # Increment the error counter if the response status code is not 200
        error_counter.inc()

        # Insert the error rate into the database
        
        cursor.execute('INSERT INTO error_rates (error_rate) VALUES (?)', (1,))
        conn.commit()
        
    r = json.loads(response.text)
    # Extract prompt tokens from the API response
    # Extract prompt tokens from the API response

    prompt_tokens = r['usage']['prompt_tokens']
    print(r['usage']['prompt_tokens'])

    # Store the prompt tokens in the database
    cursor.execute('INSERT INTO prompt_tokens (prompt_tokens) VALUES (?)', (prompt_tokens,))
    conn.commit()
    conn.close()
    print('LALAAAALAALAAA')
    print(r['choices'][0]['message']['content'])
    return response.json()
    # Use the LLM model to classify text

def reset_tables():
    # Connect to the SQLite database
    conn = sqlite3.connect('response_times.db')
    cursor = conn.cursor()

     # Drop existing tables if they exist
    cursor.execute('DROP TABLE IF EXISTS response_times')
    cursor.execute('DROP TABLE IF EXISTS error_rates')
    cursor.execute('DROP TABLE IF EXISTS moderation_decisions')
    cursor.execute('DROP TABLE IF EXISTS request_volumes')

    # Create new tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS response_times (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            response_time REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS error_rates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            error_rate INTEGER
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS moderation_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            decision TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS request_volumes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()


def moderate_with_bert(text):
    result = pipeline(text)
    return result


def decide_moderation(llm_result, bert_result):
    # result = llm_result + bert_result
    return llm_result

def gather_metrics():
    # Implement code to gather usage statistics
    # Track requests, errors, processing times, etc.
    return 

if __name__ == '__main__':
    reset_tables()  # Reset tables before starting the program
    app.run(debug=True)
