import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import re
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from enforcer import Enforcer
from collections import deque

# uncomment this line for a Syntax error so this file cannot run since it auto starts a very large gb download (starcoder LLM from huggingface)

class Model:
    def __init__(self):
        model_name = "bigcode/starcoder"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, use_auth_token=True)

    def generate_code(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500)

        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        match = re.search(r"def\s+(\w+)\s*\[\s\S]*?pdk:\s+\w+", generated_code)
        if match:
            function_name = match.group(1)
        else:
            raise ValueError("Function name not found in generated code (Model.generate_code)")

        required_line = [
            "from glayout.flow.pdk.sky130_mapped import sky130_mapped_pdk",
            f"{function_name}_cell.write_gds('{function_name}.gds')"
        ]

        if ".write_gds" not in generated_code:
            generated_code += "\n" + "\n".join(required_line)


        return generated_code
    
    def save_model(self, agent, filename="rl_model.pth"):
        torch.save(agent.state_dict(), filename)
        print(f"Model saved to {filename}")

    def load_model(self, agent, filename="rl_model.pth"):
        if os.path.exists(filename):
            agent.load_state_dict(torch.load(filename))
            print(f"Model loaded from {filename}")
        else:
            print(f"Model file {filename} not found")

class CircuitEnvironment:
    def __init__(self, action_size):
        self.action_size = action_size
        self.enforcer = Enforcer()
        self.glayout_output_folder = "glayout_output_folder"
        self.gds_output_folder = "gds_output_folder"
        self.agent = Model()
        if not os.path.exists(self.glayout_output_folder):
            os.makedirs(self.glayout_output_folder, exist_ok=True)
        if not os.path.exists(self.gds_output_folder):
            os.makedirs(self.gds_output_folder, exist_ok=True)

    def reset(self):
        
        self.state = np.random.rand(10)
        """
         the 10 represents 10 dimension vector (each dimension is a parameter such as width)
         This has to be changed to some sort of list that has parameters WITH VALUES
        Right now the randomis a palceholder for states
        """
        
        return self.state
    
    def step(self, action):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        glayout_file = f"{self.glayout_output_folder}/circuit_{timestamp}.py"
        self.generate_glayout(glayout_file, action)

        try:
            subprocess.run(['python',glayout_file], check=True)
        except Exception as e:
            raise ValueError(f"Error executing {glayout_file}: {e}")
        

        gds_file = f"{self.gds_output_folder}/circuit_{timestamp}.gds"



        


        errors = self.enforcer.run_drc_check(gds_file) # gds_file is a placeholder for the gds_file generated from running the glayout code of the LLM
        reward = self.enforcer.reward(errors)

        next_state = np.random.rand(10) # Placeolder. Update this to reflect a circuit with circuit parameters
        """
        Same point as above
        The random.rand is a placeholder
        next_state should have the same parameters as state but with different values that are updated based on learning (maybe what type of error)
        """
        done = errors == 0 or reward == 1
        return next_state, reward, done

    def generate_glayout(self, glayout_file, action):

        prompt = f"generate a circuit for action {action}" 
        """
        This is a placeholder prompt that right now prompts for a complete circuit gneeration
        Using a context file and training, this prompt should be changed so it can take prompts about existing circuits for cusotmization
        """

        glayout_code = self.agent.generate_code(prompt)
        with open(glayout_file, "w") as f:
            f.write(glayout_code)


def train_rl_model(episodes):

        

    environment = CircuitEnvironment(10) 

    environment.agent.load_model(environment.agent)



    """
    Right now, the model is made with a finite action space of 10 (10 is a placeholder)
    Determine if a finite or infinite space is best for the action space and change
    If sticking with finite, change 10 so it reflects the true number of actions
    Ex:
        Width can be +1, -1, or 0 --- 3 actions
            + 0 or not changing a parameter counts as an action
    total actions = (num of actions per parameter) all multiplied together
    """



    for episode in range(episodes):
        environment.reset()
        total_reward = 0
        done = False


        while not done:

            action = random.randint(0, environment.action_size - 1)

            """
                This action has to be updated so it randomly chooses one of the possible actions/states
            """

            next_state, reward, done = environment.step(action)
            total_reward += reward
        
        print(f"Episode {episode + 1} reward: {reward} total_reward: {total_reward}")

        environment.agent.save_model(environment.agent)

    print("Training complete")


# train_rl_model(100) # Placeholder for number of episodes to train the model



"""
Todo:

2/16
Make a function or seperate file maybe for running the model without training it

"""
       

