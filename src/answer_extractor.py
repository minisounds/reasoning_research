import os
from openai import OpenAI

api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI()

def find_answer(model_response): 
    prompt = f"""
    You are a precise answer extractor. Given a response to a test problem (which includes the test problem in the beginning), your task is to extract ONLY the final numerical answer. Do not include any units, explanations, or additional text. DO NOT SOLVE THE PROBLEM YOURSELF. If there are multiple numbers in the response, identify and return only the final answer. If no clear numerical answer is found, return -1758.

    Response to analyze:
    {model_response}

    Extract the final numerical answer from the above response. Your output should be ONLY the number, or None if no clear numerical answer is found.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts numerical answers from text."},
            {"role": "user", "content": prompt}
        ]
    )

    extracted_answer = response.choices[0].message.content
    
    return extracted_answer

def mmlu_find_answer_gpt(model_response): 
    prompt = f"""
    You are a precise answer extractor. Given a response to a multiple choice problem, your task is to extract ONLY the final answer choice the model makes (either A, B, C, or D). Do not include any units, explanations, or additional text. DO NOT SOLVE THE QUESTION AND PROVIDE THE ANSWER YOURSELF. If there are multiple answers in the response, identify and return only the final answer. If no clear answer choice is found, return 'None'.

    Response to analyze:
    {model_response}

    Extract the final answer choice from the above response. Your output should be ONLY the final multiple choice answer picked in the response (either A, B, C, or D), or None if no clear answer choice is found.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts multiple choice answers from text."},
            {"role": "user", "content": prompt}
        ]
    )

    extracted_answer = response.choices[0].message.content
    
    return extracted_answer
