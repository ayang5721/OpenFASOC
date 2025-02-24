# This file contains the enforcer function for the reinforcement learning model.

# Given a gds layout 
from gdsfactory.pdk import Pdk
from glayout.flow.pdk.mappedpdk import MappedPDK
from glayout.flow.pdk.sky130_mapped import sky130_mapped_pdk
import subprocess
import os
import drcTester


class Enforcer:
    def __init__(self):
        self.drc_errors = 0
        self.drc_error_output = ""
        self.lvs_errors = 0
        self.lvs_error_output = ""
        self.pex_errors = 0
        self.pex_error_output = ""
        

    def enforce_drc(self, gds_file, cell_name):

        drc_error_output = drcTester.run_drc(gds_file, cell_name, '../../../../common/drc-lvs-check/sky130A/sky130A.magicrc')
        return drc_error_output
    
    def drc_num(self):
        self.errors = sum(1 for line in self.drc_error_output.splitlines() if 'DRC ERROR' in line)
        return self.drc_errors
        


