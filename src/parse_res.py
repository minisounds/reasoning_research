import json
import os

# Read the JSON file
with open('steered_gsm8k', 'r') as file:
    data = json.load(file)

# Process each item in the list
parsed_responses = []
for item in data:
    # Split the string by "assistant"
    parts = item.split("assistant")
    
    # If there's a part after "assistant", add it to the list
    if len(parts) > 1:
        parsed_responses.append(parts[-1].strip())

# Append the parsed responses to the existing file
with open('parsed_responses.txt', 'a') as file:
    file.write("NEW RESPONSES HERE: ")
    for response in parsed_responses:
        file.write(response + '\n\n')

# Get the number of new responses added
new_responses_count = len(parsed_responses)

# Print the result
print(f"Appended {new_responses_count} new responses to 'parsed_responses.txt'")