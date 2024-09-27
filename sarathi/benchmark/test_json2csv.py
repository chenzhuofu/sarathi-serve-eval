import pandas as pd
import json
import random
from datetime import datetime, timedelta
import math

# Load the JSON data
with open('./data/emission.json', 'r') as file:
    data = json.load(file)

# Initialize an empty list to store the converted data
csv_data = []

# Generate a random start date for the requests
start_date = datetime(2023, 9, 21)

# Function to estimate the number of tokens from a prompt
def estimate_token_count(text):
    # Estimate tokens based on word count (approximation)
    word_count = len(text.split())
    token_count = math.ceil(word_count * 1.3)  # Roughly 1.3 tokens per word
    return token_count

# Iterate over the JSON data to convert to CSV format
for i, entry in enumerate(data):
    # Estimate prompt tokens based on the length of the prompt text
    # prompt_tokens = estimate_token_count(entry["prompt"])
    prompt_tokens = entry["input_length"]

    # Completion tokens are estimated or constant; here we use slo_ratio to scale it
    completion_tokens = entry["output_length"]
    
    # Convert emission_time_ms to time in seconds
    time_in_seconds = entry["emission_time_ms"] / 1000

    # Calculate a date by adding seconds to the start date
    date = start_date + timedelta(seconds=time_in_seconds)
    
    # TODO: Calculate request expected latency 
    # baseline_latency_ms = 1
    # expected_latency = entry["slo_ratio"] * baseline_latency_ms * completion_tokens 

    # Add the processed data to the list
    csv_data.append({
        "Date": date.strftime('%Y-%m-%d'),
        "Time": time_in_seconds,  # Time since the start in seconds
        "PromptTokenCount": prompt_tokens,  # Estimated prefill tokens
        "CompletionTokenCount": completion_tokens,  # Estimated completion tokens
    })

# Create a DataFrame from the list
df = pd.DataFrame(csv_data)

# Save the DataFrame to a CSV file
df.to_csv('./data/emission_trace.csv', index=False)

print("CSV file generated successfully.")
