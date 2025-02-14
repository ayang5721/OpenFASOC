import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# SYNTax roeggorejopje 2187128 #Syntax error so this file cannot run since it auto starts a 50 gb download

# print('hello')
"""
model_name = "bigcode/starcoder"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, use_auth_token = True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, use_auth_token = True)
"""

class SimpleEnvironment:
    def __init__(self, max_number = 10):
        self.max_number = max_number
        self.target = 5
        self.state = random.randint(0, self.max_number)
    
    def step(self, action):
        error = -abs(self.target - action)
        if error == 0:
            return self.state, 10, True
        else:  
            return self.state, error, False
        
    
    def reset(self):
        self.target = 5
        self.state = random.randint(0, self.max_number)
        return self.state
    

class Qlearning:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.99):
        self.q_table = np.zeros((state_size, action_size))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.action_size = action_size

    def choose_action(self, state):
        if random.uniform(0,1) < self.exploration_rate:
            return random.choice(range(self.action_size))
        return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

    
env = SimpleEnvironment()
agent = Qlearning(state_size = 11, action_size = 11)

for ep in range(100):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.update_q_table(state, action, reward, next_state)
        state = next_state
        print("guess: ", action)
    agent.decay_exploration()



user_number = int(input("Enter a number between 0 and 10: "))
guess = agent.choose_action(user_number)
print(f"Agent guessed: {guess}")   
# print("Final Q-table: ", agent.q_table)


    