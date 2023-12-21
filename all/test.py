import requests
import json
import pandas as pd

# Define the URL of your content moderation API
API_URL = 'http://localhost:5000/moderate'  # Replace with your actual API URL

# Function to send a comment for moderation
def moderate_comment(comment_text):
    payload = {'text': comment_text}
    headers = {'Content-Type': 'application/json'}

    response = requests.post(API_URL, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        moderation_result = response.json()['moderation_result']
        return moderation_result
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

# # Load YouTube comments data from a file or dataset
# # Replace this with your actual data loading code
# comments_data = [
#     "This is a great video lil bitch!",
#     "You guys are awesome!",
#     "This is the worst video ever!",
#     "You should be banned for this.",
#     "I hate this channel."
# ]



# Load YouTube comments data from a CSV file
comments_data = pd.read_csv('../archive/youtoxic_english_1000.csv')['Text'][0:1000].tolist()


# Test the API with the loaded comments
for comment in comments_data:
    moderation_result = moderate_comment(comment)
    print(moderation_result)
    print("ALPER")
    if moderation_result is None:
        # print(f"Comment: {comment}")
        print(f"Moderation Result: {moderation_result}")
        print()

# You can analyze the moderation results as needed
