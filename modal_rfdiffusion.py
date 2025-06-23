# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
Currently not working due to library incompatibilities.



# Instructions
Use contigs to define continious chains.
Use a : to define multiple contigs and a / to define mutliple segments within a contig.
For example:

## unconditional
contigs='100' - diffuse monomer of length 100
contigs='50:100' - diffuse hetero-oligomer of lengths 50 and 100
contigs='50' symmetry='cyclic' order=2 - make two copies of the defined contig(s) and add a symmetry constraint, for homo-oligomeric diffusion.

## binder design
contigs='A:50' pdb='4N5T' - diffuse a binder of length 50 to chain A of defined PDB.
contigs='E6-155:70-100' pdb='5KQV' hotspot='E64,E88,E96' - diffuse a binder of length 70 to 100 (sampled randomly) to chain E and defined hotspot(s).

## motif scaffolding
contigs='40/A163-181/40' pdb='5TPN'
contigs='A3-30/36/A33-68' pdb='6MRR' - diffuse a loop of length 36 between two segments of defined PDB ranges.

## partial diffusion
contigs='' pdb='6MRR' - noise all coordinates
contigs='A1-10' pdb='6MRR' - keep first 10 positions fixed, noise the rest
contigs='A' pdb='1SSC' - fix chain A, noise the rest

## hints and tips
pdb='' leave blank to get an upload prompt
contigs='50-100' use dash to specify a range of lengths to sample from

e.g., to make a binder for 1A00
modal run modal_rfdiffusion.py --pdb 1A00 --contigs "A,B:20"
"""

import glob
import os

from subprocess import run
from pathlib import Path

from modal import Image, App

OUTPUT_ROOT = "rfdiffusion"
GPU = os.environ.get("GPU", "A10G")
TIMEOUT = int(os.environ.get("TIMEOUT", 15))

app = App()


image = (
    Image
    # .micromamba(python_version="3.12")
    .from_registry("nvidia/cuda:12.1.1-runtime-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "aria2", "build-essential", "cmake")
    .run_commands(
        "mkdir params;"
        "aria2c -q -x 16 https://files.ipd.uw.edu/krypton/schedules.zip;"
        "aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt;"
        "aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt;"
        "aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar;"
        "tar -xf alphafold_params_2022-12-06.tar -C params;"
        "touch params/done.txt;"
    )
    .run_commands("git clone https://github.com/sokrypton/RFdiffusion.git")
    .pip_install(
        ["torch==2.6.0+cu124", "torchvision==0.21.0+cu124", "torchdata==0.10.0"],
        index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        ["dgl==2.1.0+cu121"], find_links="https://data.dgl.ai/wheels/cu121/repo.html"
    )
    .pip_install(
        [
            "jedi",
            "omegaconf",
            "hydra-core",
            "icecream",
            "pyrsistent",
            "pynvml",
            "decorator",
            "psutil",
        ]
    )
    .pip_install("git+https://github.com/NVIDIA/dllogger#egg=dllogger")
    .pip_install(["tqdm", "pydantic"])
    .run_commands("cd RFdiffusion/env/SE3Transformer; pip install .")
    .run_commands(
        "wget -qnc https://files.ipd.uw.edu/krypton/ananas;" "chmod +x ananas"
    )
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign"
    )
    .run_commands(
        "mkdir RFdiffusion/models;"
        "mv Base_ckpt.pt RFdiffusion/models;"
        "mv Complex_base_ckpt.pt RFdiffusion/models;"
        "unzip schedules.zip;"
        "rm schedules.zip;"
    )
    .run_commands("mv /RFdiffusion/* /root")
    .env({"DGLBACKEND": "pytorch"})
)

# ColabDesign imports
with image.imports():
    import os
    import json
    import random
    import signal
    import string
    import time
    import numpy as np


def get_pdb(pdb_input):
    """Get PDB file either from local file or download by code.

    Args:
        pdb_input (str): Either a local file path or PDB/UniProt code

    Returns:
        str: Path to the PDB file
    """
    if os.path.isfile(pdb_input):
        return pdb_input
    elif len(pdb_input) == 4:
        if not os.path.isfile(f"{pdb_input}.pdb1"):
            os.system(f"wget -qnc https://files.rcsb.org/download/{pdb_input}.pdb1.gz")
            os.system(f"gunzip {pdb_input}.pdb1.gz")
        return f"{pdb_input}.pdb1"
    else:
        os.system(
            f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_input}-F1-model_v3.pdb"
        )
        return f"AF-{pdb_input}-F1-model_v3.pdb"


def run_ananas(pdb_str, path, sym=None):
    """AnAnaS : software for analytical analysis of symmetries in protein structures
    https://hal.science/hal-02931690/document
    """
    from colabdesign.rf.utils import sym_it

    pdb_filename = f"{OUTPUT_ROOT}/{path}/ananas_input.pdb"
    out_filename = f"{OUTPUT_ROOT}/{path}/ananas.json"
    with open(pdb_filename, "w") as handle:
        handle.write(pdb_str)

    cmd = f"./ananas {pdb_filename} -u -j {out_filename}"
    if sym is None:
        os.system(cmd)
    else:
        os.system(f"{cmd} {sym}")

    # parse results
    try:
        out = json.loads(open(out_filename, "r").read())
        results, AU = out[0], out[-1]["AU"]
        group = AU["group"]
        chains = AU["chain names"]
        rmsd = results["Average_RMSD"]
        print(f"AnAnaS detected {group} symmetry at RMSD:{rmsd:.3}")

        C = np.array(results["transforms"][0]["CENTER"])
        A = [np.array(t["AXIS"]) for t in results["transforms"]]

        # apply symmetry and filter to the asymmetric unit
        new_lines = []
        for line in pdb_str.split("\n"):
            if line.startswith("ATOM"):
                chain = line[21:22]
                if chain in chains:
                    x = np.array([float(line[i : (i + 8)]) for i in [30, 38, 46]])
                    if group[0] == "c":
                        x = sym_it(x, C, A[0])
                    if group[0] == "d":
                        x = sym_it(x, C, A[1], A[0])
                    coord_str = "".join(["{:8.3f}".format(a) for a in x])
                    new_lines.append(line[:30] + coord_str + line[54:])
            else:
                new_lines.append(line)
        return results, "\n".join(new_lines)

    except:
        return None, pdb_str


def run_inference(command, steps, num_designs=1, visual="none"):
    def run_command_and_get_pid(command):
        pid_file = "/dev/shm/pid"
        os.system(f"nohup {command} > /dev/null & echo $! > {pid_file}")
        with open(pid_file, "r") as f:
            pid = int(f.read().strip())
        os.remove(pid_file)
        return pid

    def is_process_running(pid):
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        else:
            return True

    # clear previous run
    for n in range(steps):
        if os.path.isfile(f"/dev/shm/{n}.pdb"):
            os.remove(f"/dev/shm/{n}.pdb")

    pid = run_command_and_get_pid(command)
    try:
        fail = False
        for _ in range(num_designs):
            # for each step check if output generated
            for n in range(steps):
                wait = True
                while wait and not fail:
                    time.sleep(0.1)
                    if os.path.isfile(f"/dev/shm/{n}.pdb"):
                        pdb_str = open(f"/dev/shm/{n}.pdb").read()
                        if pdb_str[-3:] == "TER":
                            wait = False
                        elif not is_process_running(pid):
                            fail = True
                    elif not is_process_running(pid):
                        fail = True

                if fail:
                    break

                else:
                    if visual != "none":
                        pass
                        # with run_output:
                        #    run_output.clear_output(wait=True)
                        #    if visual == "image":
                        #        xyz, bfact = get_ca(f"/dev/shm/{n}.pdb", get_bfact=True)
                        #        fig = plt.figure()
                        #        fig.set_dpi(100);fig.set_figwidth(6);fig.set_figheight(6)
                        #        ax1 = fig.add_subplot(111);ax1.set_xticks([]);ax1.set_yticks([])
                        #        plot_pseudo_3D(xyz, c=bfact, cmin=0.5, cmax=0.9, ax=ax1)
                        #        plt.show()
                        #    if visual == "interactive":
                        #        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
                        #        view.addModel(pdb_str,'pdb')
                        #        view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':0.5,'max':0.9}}})
                        #        view.zoomTo()
                        #        view.show()
                if os.path.exists(f"/dev/shm/{n}.pdb"):
                    os.remove(f"/dev/shm/{n}.pdb")
            if fail:
                break

        while is_process_running(pid):
            time.sleep(0.1)

    except KeyboardInterrupt:
        os.kill(pid, signal.SIGTERM)


def run_diffusion(
    contigs,
    path,
    pdb=None,
    iterations=50,
    symmetry="none",
    order=1,
    hotspot=None,
    chains=None,
    add_potential=False,
    num_designs=1,
    visual="none",
):
    from inference.utils import parse_pdb
    from colabdesign.rf.utils import get_ca
    from colabdesign.rf.utils import fix_contigs, fix_partial_contigs, fix_pdb
    from colabdesign.shared.protein import pdb_to_string
    from colabdesign.shared.plot import plot_pseudo_3D

    full_path = f"{OUTPUT_ROOT}/{path}"
    os.makedirs(full_path, exist_ok=True)

    opts = [
        f"inference.output_prefix={full_path}",
        f"inference.num_designs={num_designs}",
    ]

    if chains == "":
        chains = None

    # determine symmetry type
    if symmetry in ["auto", "cyclic", "dihedral"]:
        if symmetry == "auto":
            sym, copies = None, 1
        else:
            sym, copies = {
                "cyclic": (f"c{order}", order),
                "dihedral": (f"d{order}", order * 2),
            }[symmetry]
    else:
        symmetry = None
        sym, copies = None, 1

    # determine mode
    contigs = contigs.replace(",", " ").replace(":", " ").split()
    is_fixed, is_free = False, False
    fixed_chains = []

    for contig in contigs:
        for x in contig.split("/"):
            a = x.split("-")[0]
            if a[0].isalpha():
                is_fixed = True
                if a[0] not in fixed_chains:
                    fixed_chains.append(a[0])
            if a.isnumeric():
                is_free = True

    if len(contigs) == 0 or not is_free:
        mode = "partial"
    elif is_fixed:
        mode = "fixed"
    else:
        mode = "free"

    #
    # fix input contigs
    #
    if mode in ["partial", "fixed"]:
        pdb_str = pdb_to_string(get_pdb(pdb), chains=chains)
        print("pdb_str:", pdb_str[:1000])
        if symmetry == "auto":
            a, pdb_str = run_ananas(pdb_str, path)
            if a is None:
                print(f"ERROR: no symmetry detected")
                symmetry = None
                sym, copies = None, 1
            else:
                if a["group"][0] == "c":
                    symmetry = "cyclic"
                    sym, copies = a["group"], int(a["group"][1:])
                elif a["group"][0] == "d":
                    symmetry = "dihedral"
                    sym, copies = a["group"], 2 * int(a["group"][1:])
                else:
                    print(
                        f'ERROR: the detected symmetry ({a["group"]}) not currently supported'
                    )
                    symmetry = None
                    sym, copies = None, 1

        elif mode == "fixed":
            pdb_str = pdb_to_string(pdb_str, chains=fixed_chains)

        pdb_filename = f"{full_path}/input.pdb"
        with open(pdb_filename, "w") as handle:
            handle.write(pdb_str)

        parsed_pdb = parse_pdb(pdb_filename)
        opts.append(f"inference.input_pdb={pdb_filename}")
        if mode in ["partial"]:
            iterations = int(80 * (iterations / 200))
            opts.append(f"diffuser.partial_T={iterations}")
            contigs = fix_partial_contigs(contigs, parsed_pdb)
        else:
            opts.append(f"diffuser.T={iterations}")
            contigs = fix_contigs(contigs, parsed_pdb)
    else:
        assert mode == "free"
        opts.append(f"diffuser.T={iterations}")
        parsed_pdb = None
        contigs = fix_contigs(contigs, parsed_pdb)

    if hotspot is not None and hotspot != "":
        opts.append(f"ppi.hotspot_res=[{hotspot}]")

    # setup symmetry
    if sym is not None:
        sym_opts = ["--config-name symmetry", f"inference.symmetry={sym}"]
        if add_potential:
            sym_opts += [
                "'potentials.guiding_potentials=[\"type:olig_contacts,weight_intra:1,weight_inter:0.1\"]'",
                "potentials.olig_intra_all=True",
                "potentials.olig_inter_all=True",
                "potentials.guide_scale=2",
                "potentials.guide_decay=quadratic",
            ]
        opts = sym_opts + opts
        contigs = sum([contigs] * copies, [])

    opts.append(f"'contigmap.contigs=[{' '.join(contigs)}]'")
    opts += ["inference.dump_pdb=True", "inference.dump_pdb_path='/dev/shm'"]

    print("mode:", mode)
    print("output:", full_path)
    print("contigs:", contigs)

    opts_str = " ".join(opts)
    cmd = f"./run_inference.py {opts_str}"
    print(cmd)

    # inference step
    run_inference(cmd, iterations, num_designs, visual=visual)

    # Fix pdbs
    for n in range(num_designs):
        pdbs = [
            f"{OUTPUT_ROOT}/traj/{path}_{n}_pX0_traj.pdb",
            f"{OUTPUT_ROOT}/traj/{path}_{n}_Xt-1_traj.pdb",
            f"{full_path}_{n}.pdb",
        ]
        for pdb in pdbs:
            with open(pdb, "r") as handle:
                pdb_str = handle.read()
            with open(pdb, "w") as handle:
                handle.write(fix_pdb(pdb_str, contigs))

    return contigs, copies


def designability_test(
    contigs,
    path,
    copies,
    num_designs,
    num_seqs: int = 8,
    initial_guess: bool = False,
    num_recycles: int = 1,
    use_multimer: bool = False,
    rm_aa: str = "",
    mpnn_sampling_temp: float = 0.1,
):
    """run ProteinMPNN to generate a sequence and AlphaFold to validate
    @markdown - for **binder** design, we recommend `initial_guess=True num_recycles=3`
    """
    # @title run **ProteinMPNN** to generate a sequence and **AlphaFold** to validate
    # num_seqs = 8 #@param ["1", "2", "4", "8", "16", "32", "64"] {type:"raw"}
    # initial_guess = False #@param {type:"boolean"}
    # num_recycles = 1 #@param ["0", "1", "2", "3", "6", "12"] {type:"raw"}
    # use_multimer = False #@param {type:"boolean"}
    # rm_aa = "C" #@param {type:"string"}
    # mpnn_sampling_temp = 0.1 #@param ["0.0001", "0.1", "0.15", "0.2", "0.25", "0.3", "0.5", "1.0"] {type:"raw"}
    # @markdown - for **binder** design, we recommend `initial_guess=True num_recycles=3`

    if not os.path.isfile("params/done.txt"):
        # TEMPTEMP checks for alphafold download so skippable
        pass

    print("downloading AlphaFold params...")
    while not os.path.isfile("params/done.txt"):
        time.sleep(5)

    contigs_str = ":".join(contigs)
    opts = [
        f"--pdb={OUTPUT_ROOT}/{path}_0.pdb",
        f"--loc={OUTPUT_ROOT}/{path}",
        f"--contig={contigs_str}",
        f"--copies={copies}",
        f"--num_seqs={num_seqs}",
        f"--num_recycles={num_recycles}",
        f"--rm_aa={rm_aa}",
        f"--mpnn_sampling_temp={mpnn_sampling_temp}",
        f"--num_designs={num_designs}",
    ]
    if initial_guess:
        opts.append("--initial_guess")
    if use_multimer:
        opts.append("--use_multimer")
    opts = " ".join(opts)
    run(["python", "colabdesign/rf/designability_test.py", opts], check=True)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT * 60)
def rfdiffusion(
    contigs: str,
    pdb_content: bytes,
    pdb_name: str,
    is_pdb_id: bool,
    iterations: int = 25,
    hotspot: str = "",
    num_designs: int = 1,
    visual: str = "image",
    symmetry: str = "none",
    order: int = 1,
    chains: str = "",
    add_potential: bool = True,
    name: str = None,
):
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td_in:
        if is_pdb_id:
            # Use the PDB ID directly for get_pdb function
            pdb_input = pdb_name
        else:
            # Write PDB content to temporary file
            temp_pdb_file = Path(td_in) / "input.pdb"
            if pdb_content:
                temp_pdb_file.write_bytes(pdb_content)
            pdb_input = str(temp_pdb_file)

        name = name or pdb_name

        # determine where to save
        path = name
        while os.path.exists(f"{OUTPUT_ROOT}/{path}_0.pdb"):
            path = (
                name
                + "_"
                + "".join(random.choices(string.ascii_lowercase + string.digits, k=5))
            )

        flags = {
            "contigs": contigs,
            "pdb": pdb_input,
            "order": order,
            "iterations": iterations,
            "symmetry": symmetry,
            "hotspot": hotspot,
            "path": path,
            "chains": chains,
            "add_potential": add_potential,
            "num_designs": num_designs,
            "visual": visual,
        }

        for k, v in flags.items():
            if isinstance(v, str):
                flags[k] = v.replace("'", "").replace('"', "")

        run_diffusion(**flags)

        # designability test here

        return [
            (outfile, open(outfile, "rb").read())
            for outfile in glob.glob(f"{OUTPUT_ROOT}/**/*.*", recursive=True)
            if os.path.isfile(outfile)
        ]


@app.local_entrypoint()
def main(
    pdb: str,
    contigs: str,
    name: str = "",
    iterations: int = 25,
    hotspot: str = "",
    num_designs: int = 1,
    visual: str = "image",
    symmetry: str = "none",
    order: int = 1,
    chains: str = "",
    add_potential: bool = True,
    run_name: str = None,
    out_dir: str = "./out/rfdiffusion",
):
    """Local entrypoint to run RFdiffusion using Modal.

    Args:
        pdb (str): PDB code, UniProt code, or path to a PDB file.
        contigs (str): Contigs specification for RFdiffusion.
        name (str): Name for the run. Defaults to PDB name.
        iterations (int): Number of iterations. Defaults to 25.
        hotspot (str): Hotspot residues. Defaults to "".
        num_designs (int): Number of designs. Defaults to 1.
        visual (str): Visualization mode. Defaults to "image".
        symmetry (str): Symmetry type. Defaults to "none".
        order (int): Symmetry order. Defaults to 1.
        chains (str): Chain selection. Defaults to "".
        add_potential (bool): Add potential. Defaults to True.
        run_name (str | None): Optional name for the run, used for organizing output files.
        out_dir (str): Directory to save the output files. Defaults to "./out/rfdiffusion".
    """
    from datetime import datetime

    # Check if input is a file path or PDB ID
    pdb_path = Path(pdb)
    if pdb_path.exists():
        # Local file - read content and pass as bytes
        pdb_content = pdb_path.read_bytes()
        pdb_name = pdb_path.stem
        is_pdb_id = False
    elif len(pdb) in [4, 5] and pdb.replace("-", "").isalnum():
        # Looks like a PDB ID or UniProt ID - pass as string to remote function
        pdb_content = None
        pdb_name = pdb
        is_pdb_id = True
    else:
        raise FileNotFoundError(
            f"PDB file not found and '{pdb}' doesn't look like a valid PDB/UniProt ID: {pdb}"
        )

    outputs = rfdiffusion.remote(
        contigs=contigs,
        pdb_content=pdb_content or b"",
        pdb_name=pdb_name,
        is_pdb_id=is_pdb_id,
        iterations=iterations,
        hotspot=hotspot,
        num_designs=num_designs,
        visual=visual,
        symmetry=symmetry,
        order=order,
        chains=chains,
        add_potential=add_potential,
        name=name,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]
    out_dir_full = Path(out_dir) / (run_name or today)

    for out_file, out_content in outputs:
        out_path = out_dir_full / out_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_path, "wb") as out:
                out.write(out_content)
