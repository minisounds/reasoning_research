import random

# create function to generate random 3 digit addition
def generate_addition_problem(): 
    num1 = random.randint(1, 500)
    num2 = random.randint(1, 500)
    # num3 = random.randint(1, 1000)
    # answer = (num1 + num2) * num3
    # question = f"({num1}+{num2})*{num3}"
    question = f"{num1}+{num2}"
    answer = num1 + num2
    return question, answer
