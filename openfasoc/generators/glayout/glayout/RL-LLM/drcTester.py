from os import environ, path, makedirs
import shutil
import subprocess

"""
# Set up environment variables (modify paths as needed)
os.environ["PDK_ROOT"] = sky130_mapped_pdk
os.environ["MAGIC"] = "/usr/local/bin/magic"  # Adjust if Magic is installed elsewhere
"""

def run_drc(
        gds_path: str,
        cell_name: str,
        pdk_magicrc_path: str,
        runs_dir_path: str = "drc_runs"
) -> str:
    """
    Runs DRC on a given GDS and returns the report text.
    Arguments:
        - `gds_path`: Path to the GDS file to test
        - `cell_name`: Name of the cell in the GDS to run DRC on
        - `pdk_magicrc_file`: Path to the `.magicrc` file
    """
    # Create a clean directory to run DRC inside
    if not path.exists(runs_dir_path):
        makedirs(runs_dir_path)
    else:
        shutil.rmtree(runs_dir_path)
        makedirs(runs_dir_path)

    shutil.copy(gds_path, path.join(runs_dir_path, 'test.gds'))

    commands_file_path = path.join(runs_dir_path, 'run_drc.tcl')
    shutil.copy(
        path.join(path.dirname(__file__), 'run_drc.tcl'), commands_file_path
    )

    with open(path.join(runs_dir_path, 'magic_drc.log'), 'w') as logfile:
        p = subprocess.run(
            f"bash -c 'CELL_NAME={cell_name} magic -noconsole -rcfile {path.abspath(pdk_magicrc_path)} -dnull run_drc.tcl < /dev/null'",
            cwd=runs_dir_path,
            shell=True,
            check=True,
            stdout=logfile,
            stderr=logfile
        )

        print()

    return open(path.join(runs_dir_path, 'drc.rpt')).read()

if __name__ == "__main__":
    report = run_drc('ChargePump.gds', 'ChargePump', '../../../../common/drc-lvs-check/sky130A/sky130A.magicrc')
    print("DRC complete.")
    print(report)
