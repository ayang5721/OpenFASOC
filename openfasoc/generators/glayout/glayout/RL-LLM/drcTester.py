import os
import subprocess
from glayout.flow.pdk.sky130_mapped import sky130_mapped_pdk

"""
# Set up environment variables (modify paths as needed)
os.environ["PDK_ROOT"] = sky130_mapped_pdk
os.environ["MAGIC"] = "/usr/local/bin/magic"  # Adjust if Magic is installed elsewhere
"""
# Paths to layout file and rules
layout_file = "ChargePump.gds"
drc_output = "drc.out"
drc_rules = "/path/to/sky130A.tech"

# Run Magic DRC
def run_drc():
    cmd = f"""
    {os.environ['MAGIC']} -dnull -noconsole << EOF
    gds read {layout_file}
    load topcell
    select top cell
    drc check
    drc catchup
    drc count
    drc save {drc_output}
    quit
    EOF
    """
    print("Running DRC...")
    subprocess.run(cmd, shell=True, check=True)
    print(f"DRC check completed. Results saved in {drc_output}")

if __name__ == "__main__":
    run_drc()
