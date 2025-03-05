from os import path, makedirs
import shutil
import subprocess

def run_pex(
    gds_path: str,
    cell_name: str,
    pdk_magicrc_file: str,
    full_extraction: bool = False,
    flatten: bool = False,
    runs_dir_path: str = "pex_runs"
) -> str:
    """
    Runs PEX on a given GDS and returns the path to the extracted netlist.
    Arguments:
        - `gds_path`: Path to the GDS file
        - `cell_name`: Name of the cell to extract
        - `pdk_magicrc_file`: Path to the `.magicrc` file
        - `full_extraction`: Whether to run full RC extraction
        - `flatten`: Whether to flatten the netlist
    """

    # Create a clean directory to run PEX inside
    if not path.exists(runs_dir_path):
        makedirs(runs_dir_path)
    else:
        shutil.rmtree(runs_dir_path)
        makedirs(runs_dir_path)

    shutil.copy(gds_path, path.join(runs_dir_path, 'pex.gds'))
    commands_file_path = path.join(runs_dir_path, 'run_pex.tcl')

    pex_script = f"""
    gds flatglob *\$\$*
    gds read pex.gds

    {f'flatten {cell_name}' if flatten else ''}
    load {cell_name}
    select top cell

    extract do local
    extract all

    {'ext2sim labels on' if full_extraction else ''}
    {'ext2sim' if full_extraction else ''}
    {'extresist tolerance 20' if full_extraction else ''}
    {'extresist' if full_extraction else ''}

    ext2spice lvs
    {'ext2spice cthresh 0' if full_extraction else ''}
    {'ext2spice extresist on' if full_extraction else ''}

    ext2spice -o extracted.spice
    exit
    """

    with open(commands_file_path, 'w') as commands_file:
        commands_file.write(pex_script)

    
    with open(path.join(runs_dir_path, 'magic_pex.log'), 'w') as logfile:
        p = subprocess.run(
            f"magic -noconsole -rcfile {path.abspath(pdk_magicrc_path)} -dnull run_pex.tcl < /dev/null'",
            cwd=runs_dir_path,
            shell=True,
            check=True,
            stdout=logfile,
            stderr=logfile
        )

    return path.join(runs_dir_path, 'extracted.spice')

def run_lvs(
        extracted_netlist: str,
        schematic_netlist: str,
        cell_name: str,
        pdk_netgen_setup_file: str,
        runs_dir_path: str = "lvs_runs"
) -> str:
    """
    Runs LVS on a given schematic and an extracted netlist and returns the report text.
    Arguments:
        - `extracted_netlist`: Path to the extracted netlist file
        - `schematic_netlist`: Path to the schematic netlist file
        - `cell_name`: The name of the cell to run LVS on
        - `pdk_netgen_setup_file`: Path to the netgen setup tcl file
    """

    # Create a clean directory to run LVS inside 
    if not path.exists(runs_dir_path):
        makedirs(runs_dir_path)
    else:
        shutil.rmtree(runs_dir_path)
        makedirs(runs_dir_path)
   
    shutil.copy(extracted_netlist, path.join(runs_dir_path, 'extracted.spice'))
    shutil.copy(schematic_netlist, path.join(runs_dir_path, 'schematic.spice'))

    with open(path.join(runs_dir_path, 'netgen_lvs.log'), 'w') as logfile:
        p = subprocess.run(
            f"netgen -batch lvs 'extracted.spice {cell_name}' 'schematic.spice {cell_name}' {pdk_netgen_setup_file} lvs.rpt",
            cwd=runs_dir_path,
            shell=True,
            check=True,
            stdout=logfile,
            stderr=logfile
        )
    
    return open(path.join(runs_dir_path, 'lvs.rpt')).read()

if __name__ == "__main__":
    extracted_netlist = run_drc('ChargePump.gds', 'ChargePump', '../../../../common/drc-lvs-check/sky130A/sky130A_setup.tcl')
    print("PEX complete.")


