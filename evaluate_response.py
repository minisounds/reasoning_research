import os
from openai import OpenAI
api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI()

def grade_response(model_response, original_question):
    prompt = f"""
    You are a 
    Original question: {original_question}
    
    Model response: {model_response}
    
    Please evaluate the quality of the model's chain of thought reasoning in its response. 
    Focus on how well the response demonstrates step-by-step reasoning to arrive at its conclusion.
    Grade it on a scale of 1-5, where:
    1 = Very Poor (No clear steps in reasoning, jumps to conclusions)
    2 = Poor (No structured attempt at step-by-step thinking, but some elaboration)
    3 = Good (Clear steps in reasoning, but some minor logical leaps)
    4 = Very Good (Well-structured chain of thought with clear, logical steps)
    5 = Excellent (Exemplary step-by-step reasoning, each point builds logically on the previous)

    Consider the following aspects in your evaluation:
    1. Clarity of each step
    2. Logical flow from one step to the next
    3. Completeness of the reasoning chain
    4. Relevance of each step to the original question
    5. Absence of unjustified leaps in logic

    Provide your grade as a single number. Do not elaborate further.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with evaluating the quality of chain of thought reasoning in responses to questions."},
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.choices[0].message.content

# Example usage
# original_question = "Three friends, Alice, Bob, and Charlie, are sitting in a row. Alice is not sitting next to Bob. Bob is sitting to the right of Charlie. Who is sitting in the middle? "
# model_response = "Let's start by analyzing the given information: 1. Alice is not sitting next to Bob. 2. Bob is sitting to the right of Charlie. From statement 1, we can conclude that either Alice is sitting on the left of Bob or on the right of Bob. But since Bob is sitting to the right of Charlie (statement 2), it means that Alice cannot be sitting on the right of Bob, otherwise, Bob would not be sitting to the right of Charlie. So, Alice must be sitting on the left of Bob. This means that Charlie is sitting on the left of Bob, and Bob is sitting on the left of Alice. Now, let's find out who is sitting in the middle. Since Charlie is sitting on the left of Bob, and Bob is sitting on the left of Alice, it means that Charlie is sitting in the middle. Therefore, the answer is Charlie. Final Answer: The final answer is Charlie. I hope it is correct."

# result = grade_response(model_response, original_question)
# print(result)