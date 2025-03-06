import torch
import random
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import re
import subprocess
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from enforcer import Enforcer
from collections import deque

# INFO uncomment this line for a Syntax error so this file cannot run since it auto starts a very large gb download (starcoder LLM from huggingface)

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size = 64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Model:
    def __init__(self):

        # find a way to make sure starcoder downloads to google cloud machine
        model_name = "bigcode/starcoder"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True, use_auth_token = True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, use_auth_token = True)

    def generate_code(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=500)

        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        match = re.search(r"def\s+(\w+)\s*\(.*?pdk\s*:\s*\w+", generated_code)
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
    
    def generate_modified_code(self, circuit, prompt):
        """
        Generates modified code based on an existing circuit and a prompt.

        Args:
            circuit (str): The existing circuit code as a string.
            prompt (str): The instruction for modification.

        Returns:
            str: The modified circuit code.
        """
        full_prompt = f"""
        Modify the following circuit code based on this instruction: '{prompt}'

        Circuit Code:
        {circuit}

        Provide the complete modified circuit code:
        """
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
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
        

    def save_checkpoint(self, q_network, optimizer, epsilon, episode, filename="rl_checkpoint.pth"):
        torch.save({
            'q_network_state_dict': q_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epsilon': epsilon,
            'episode': episode
        }, filename)
        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, q_network, optimizer, filename="rl_checkpoint.pth"):
        epsilon, episode = 0.1, 0
        if os.path.exists(filename):
            checkpoint = torch.load(filename)
            q_network.load_state_dict(checkpoint['q_network_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epsilon = checkpoint.get('epsilon', 0.1)
            episode = checkpoint.get('episode', 0)
            print(f"Checkpoint loaded from {filename}")
        else:
            print("No checkpoint found")
        
        return epsilon, episode

    

class CircuitEnvironment:
    def __init__(self):
        self.enforcer = Enforcer()
        self.glayout_output_folder = "glayout_training_output_folder"
        self.gds_output_folder = "gds_output_folder"
        self.agent = Model()
        self.state_size = 10 # Placeholder for the number of parameters in the state
        self.action_size = 10 # Placeholder for the number of actions

        self.q_network = QNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)

        self.epsilon, self.episode = self.agent.load_checkpoint(self.q_network, self.optimizer)

        self.memory = deque(maxlen=1000)
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95

        if not os.path.exists(self.glayout_output_folder):
            os.makedirs(self.glayout_output_folder, exist_ok=True)
        if not os.path.exists(self.gds_output_folder):
            os.makedirs(self.gds_output_folder, exist_ok=True)

    def reset(self):
        
        self.state = np.random.rand(10)
        """
         the 10 represents 10 dimension vector (each dimension is a parameter such as width)
         This has to be changed to some sort of list that has parameters WITH VALUES
        Right now the random is a placeholder for states
        """

        #Check to make sure this is correct state
        self.state = {
            "nmos_width": 1,
            "pmos_width": 1,
            "nmos_length": 1,
            "pmos_length": 1,
            "nmos_finger": 1,
            "pmos_finger": 1,
            "rmult": False,
            "multipliers": False,
            "with_substrate_tap": False,
            "with_tie": False,
            "with_dummy": False,
            "smart_route": False
        }
        

        return self.state
    
    def sample_action(self):
        action = {
            "operation": random.choice(["modify_width", "modify_length", "modify_finger", "add_dummy", "add_tie", "add_substrate_tap", "smart_route", "place"]),
            "component": random.choice(["nmos", "pmos"]),
            "parameter": random.choice(["width", "length", "finger"]),
            "value": np.random.uniform(-1.0, 1.0)
        }
        return action

    def select_action(self, state):
        if random.random() <= self.epsilon:
            action = self.sample_action()
            if action['operation'] in ("add_dummy", "add_tie", "add_substrate_tap", "smart_route", "place"):
                if action['value'] <= 0:
                    action['value'] = False
                else:
                    action['value'] = True
            return action 
        else:
            state_tensor = torch.FloatTensor(list(state.values())).unsqueeze(0)
            with torch.no_grad():
                    q_values = self.q_network(state_tensor)
            best_action_index = torch.argmax(q_values).item()
            return self.decode_action(best_action_index)
        
    def decode_action(self, action_index):
        operation = ["modify_width", "modify_length", "modify_finger", "add_dummy", "add_tie", "add_substrate_tap", "smart_route", "place"]
        component = ["nmos", "pmos"]
        parameter = ["width", "length", "finger"]

        #Change this so the LLM decides value, not randomness
        value = np.random.uniform(-1.0, 1.0)


        return {
            "operation": operation[action_index % len(operation)],
            "component": component[(action_index // len(operation)) % len(component)],
            "parameter": parameter[(action_index // (len(operation) * len(component))) % len(parameter)],
            "value": value
        }    
    
    def step(self, action):
        while True:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            glayout_file = f"{self.glayout_output_folder}/circuit_{timestamp}.py"
            if not os.path.exists(glayout_file):
                break

        self.generate_glayout(glayout_file, action)
        gds_file = None

        if not os.path.exists(glayout_file):
            raise FileNotFoundError("Glayout file not found (step)")

        
        syntaxError = False
        syntaxReport = None

        try:
            subprocess.run(['python',glayout_file], check=True)
        except subprocess.CalledProcessError as e:
            syntaxReport = e.output.decode("utf-8")
            syntaxError = True
            pass

        

        with open(glayout_file, "r", encoding = "utf-8") as f:
            for line in reversed(list(f)):
                if ".gds" in line:
                    gds_file = line.split(".gds")[0].split("(")[-1].strip("\"'") + ".gds"
                    break
        
        if gds_file is None:
            syntaxReport.append("GDS file not found")
            syntaxError = True
            pass


        """
        The drc/lvs/pex reports will be used as step by step context for the model to learn from
        This type of context needs to be implemented after initial pure-reward training so the model can first stabilize and avoid training errors
        Find a way to do this - gpt

        Find a way to include syntax error report as the most important error checkers since a syntax error is the most crucial error
        """
        if(syntaxError):
            syntax_report = self.enforcer.syntax_report(syntaxReport)

        drc_report = self.enforcer.enforce_drc(gds_file, f"DRC_{glayout_file.replace('.py', '.rpt')}")
        drc_errors = self.enforcer.drc_num()

        pex_report = self.enforcer.enforce_pex(gds_file, f"PEX_{glayout_file.replace('.py', '.rpt')}")
        pex_errors = self.enforcer.pex_num()

        lvs_report = self.enforcer.enforce_lvs(gds_file, f"LVS_{glayout_file.replace('.py', '.rpt')}")
        lvs_errors = self.enforcer.lvs_num()

        # Weight errors if needed
        errors = drc_errors + lvs_errors + pex_errors
        reward = - (errors ** 2)
        if errors == 0:
            reward = 1

        # INFO action update
        key = f"{action['component'].lower()}_{action['parameter']}"
        if key in self.state:
            self.state[key] += action['value']

        done = errors == 0 or reward == 1
        return self.state, reward, done

    def generate_glayout(self, glayout_file, action):

        prompt = f"generate a circuit for action {action}" 
        """
        This is a placeholder prompt that right now prompts for a complete circuit gneeration
        Using a context file and training, this prompt should be changed so it can take prompts about existing circuits for cusotmization
        """

        glayout_code = self.agent.generate_code(prompt)
        with open(glayout_file, "w") as f:
            f.write(glayout_code)

    def optimize_model(self):
        #Check this function with gpt, change this function to truly optimize (maybe using error reports?)

        if len(self.memory) < 64:
            return #INFO insufficient sample num

        batch = random.sample(self.memory, 64)
        states, actions, rewards, next_states, dones = zip(*batch) # actions is currently not used. Check if this is right or wrong

        states = torch.FloatTensor(states)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q = self.q_network(states).max(1)[0]
        with torch.no_grad():
            next_q = self.q_network(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



def train_rl_model(episodes):

    environment = CircuitEnvironment()
    episodes = environment.episode

    user_input = input (f"Do you want to reset episodes to 0? Currently episodes are {episodes}. Enter 'y' to reset or any other key to continue: ")
    if user_input.lower() == 'y':
        episodes = 0

    for episode in range(episodes):
        state = environment.reset()
        total_reward = 0
        done = False

        while not done:
            action = environment.select_action(state)
            next_state, reward, done = environment.step(action)

            environment.memory.append((list(state.values()), action, reward, list(next_state.values()), done))  
            environment.optimize_model()

            state = next_state
            total_reward += reward

        environment.epsilon = max(environment.epsilon_min, environment.epsilon * environment.epsilon_decay)
        print(f"Episode {episode + 1} Total Reward: {total_reward} Epsilon: {environment.epsilon}")

        environment.agent.save_checkpoint(
            environment.q_network,
            environment.optimizer,
            environment.epsilon,
            episode + 1
        )

    print("Training complete")

def run_model():
    
    environment = CircuitEnvironment()

    glayout_output_folder = "glayout_output_folder"
    if not os.path.exists(glayout_output_folder):
        os.makedirs(glayout_output_folder, exist_ok=True)
    
    circuit = input("Enter the path to the existing circuit code or enter n to not edit an existing circuit: ")
    prompt = input("Enter a prompt: ")

    if circuit == "n":
        glayout_code = environment.agent.generate_code(prompt)
    else:
        with open(circuit, "r") as f:
            circuit_code = f.read()
        glayout_code = environment.agent.generate_modified_code(circuit_code, prompt)

    name = input("Enter a name for the circuit or enter 1 to auto name: ")
    while True:
        if name == 1:
            time = datetime.now().strftime("%Y%m%d_%H%M%S")
            glayout_file = f"{glayout_output_folder}/circuit_{time}.py"
            if not os.path.exists(glayout_file):
                break
        else:
            glayout_file = f"{glayout_output_folder}/{name}.py"
            if os.path.exists(glayout_file):
                print("File already exists. Please enter a different name or enter 1 to auto name.") 
            else:
                break

            name = input()

    with open(glayout_file, "w") as f:
        f.write(glayout_code)

    print(f"Generated glayout code saved to {glayout_file}")


# INFO train_rl_model(100) # Placeholder for number of episodes to train the model

"""
Todo: 


        action space
        context files
        google cloud machine setup/running
       
        # Comments are things that need to be checked/fixed
"""
       

