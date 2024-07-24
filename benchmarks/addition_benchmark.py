import random

# create function to generate random 3 digit addition
def generate_addition_problem(): 
    # num1 = random.randint(1, 500)
    # num2 = random.randint(1, 500)
    # num3 = random.randint(1, 1000)
    #nums = [random.randint(1,500) for _ in range(16)]
    question = " + ".join(map(str, map(lambda _:random.randint(1,1000), range(16))))
    return question+" =", eval(question)

generate_addition_problem()