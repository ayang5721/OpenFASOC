# This file contains the enforcer function for the reinforcement learning model.

# Given a gds layout 
from gdsfactory.pdk import Pdk
from glayout.flow.pdk.mappedpdk import MappedPDK
from glayout.flow.pdk.sky130_mapped import sky130_mapped_pdk
import subprocess
import os
import drcTester
import lvs


class Enforcer:
    def __init__(self):
        self.drc_errors = 0
        self.drc_error_output = ""
        self.lvs_errors = 0
        self.lvs_error_output = ""
        self.pex_errors = 0
        self.pex_error_output = ""
        

    # EDIT ALL OF THE (ERROR)_NUM FUNCTIONS TO ACCURATELY COUNT THE NUMBER OF ERRORS IN THE OUTPUT FILE DEPENDING ON HOW THE REPORT FILES ARE FORMATTED    
    # Make sure the paths to the magic files are correct

    def enforce_drc(self, gds_file, cell_name):

        drc_error_output = drcTester.run_drc(gds_file, cell_name, '../../../../common/drc-lvs-check/sky130A/sky130A.magicrc')
        return drc_error_output
    
    def drc_num(self):
        self.errors = sum(1 for line in self.drc_error_output.splitlines() if 'DRC ERROR' in line)
        return self.drc_errors
    
    def enforce_pex(self, gds_file, cell_name):
        pex_error_output = lvs.run_pex(gds_file, cell_name, '../../../../common/drc-lvs-check/sky130A/sky130A_maigcrc')
        return pex_error_output
    
    def pex_num(self):
        self.errors = sum(1 for line in self.pex_error_output.splitlines() if 'PEX ERROR' in line)
        return self.pex_errors
    
    def enforce_lvs(self, extracted_netlist, schematic_netlist, cell_name):
        lvs_error_output = lvs.run_lvs(extracted_netlist, schematic_netlist, cell_name, 'pdk_netgen_setup_file')
        return lvs_error_output

    def lvs_num(self):
        self.errors = sum(1 for line in self.lvs_error_output.splitlines() if 'LVS ERROR' in line)
        return self.lvs_errors

        


