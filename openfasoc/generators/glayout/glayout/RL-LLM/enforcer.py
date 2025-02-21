# This file contains the enforcer function for the reinforcement learning model.

# Given a gds layout 
from gdsfactory.pdk import Pdk
from glayout.flow.pdk.mappedpdk import MappedPDK
from glayout.flow.pdk.sky130_mapped import sky130_mapped_pdk
import subprocess
import os


class Enforcer:
    def __init__(self, pdk: Pdk):
        self.pdk = pdk
        self.mapped_pdk = MappedPDK(pdk, sky130_mapped_pdk)
        self.errors = 0
        self.drc_script = "run_drc.sh"
        self.drc_output_file = "drc.out" #Make sure drc is being saved to this file from script
        # self.lvs/pex_script = 
        # self.lvs/pex_output_file =


    
    def run_drc_check(self, gds_file):
        if not os.path.exists(self.drc_script):
            raise FileNotFoundError(f"File {self.drc_script} not found")
        
        result = subprocess.run(["bash", self.drc_script, gds_file], capture_output=True, text=True)

        if result.stderr:
            raise RuntimeError(f"Error running DRC: {result.stderr}")
        
        return result #Make sure the drc output is saving to "drc.out" or some other type of file
        #result should be a file

    def drc_num(self):

        if not os.path.exists(self.drc_output_file):
            raise FileNotFoundError(f"File {self.drc_output_file} not found")
        
        errors = 0
        with open(self.drc_output_file, "r") as f:
            for line in f:
                if "Error" in line:
                    errors += 1 

        return errors
    

    # This function is incomplete since i dont know lvs/pex scripts yet for gds file
    #Also need to find spice netlist generation script from glayout.py or gds
    def run_lvs_check(self, gds_file, netlist_file):
        # finish this code

    def reward(self, errors):
        return 1 / (1 + errors)
    



    
    def enforce(self, gds_file):

        #This is essentially the main function that will be called in the RL model

        self.drc_output_file = run_drc_check(gds_file)
        self.errors = self.drc_num()
        return self.reward(self.errors)


