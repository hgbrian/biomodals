"""
Visualize a pdb file as a png.

TODO: center on the ligand, if there is one, and find the best orientation.
"""

import json
from pathlib import Path
from typing import Union

from modal import App, Image

app = App("pdb2png")

image = (
    Image.micromamba(python_version="3.11")
    .micromamba_install("pymol-open-source==2.5.0", channels=["conda-forge"])
    .apt_install("libgl1")
    .apt_install("g++")
    .pip_install(["ProDy==2.4.1"])
)


RENDER_OPTIONS = {
    "default": {
        "ray_opaque_background": "off",
        "antialias": "2",
        "orthoscopic": "on",
        "depth_cue": "0",
        "ray_trace_mode": "1",
        "ray_trace_color": "black",
    },
    "default_bw": {
        "bg_color": "white",
        "ray_opaque_background": "on",
        "antialias": "2",
        "orthoscopic": "on",
        "depth_cue": "0",
        "ray_trace_mode": "2",
    },
    "default_cartoon": {
        "ray_opaque_background": "off",
        "antialias": "2",
        "orthoscopic": "on",
        "depth_cue": "0",
        "ray_trace_mode": "3",
    },
    "dark": {
        "bg_color": "black",
        "ray_opaque_background": "on",
        "antialias": "2",
        "orthoscopic": "on",
        "light_count": "2",
        "specular": "1",
        "depth_cue": "1",
        "fog_start": "0.35",
        "ray_trace_mode": "1",
        "ray_trace_color": "black",
    },
    "flat": {
        "bg_color": "white",
        "valence": "0",
        "bg_rgb": "white",
        "reflect": "0",
        "spec_direct": "0",
        "light_count": "1",
        "spec_count": "0",
        "shininess": "0",
        "power": "1",
        "specular": "0",
        "ambient_occlusion_mode": "1",
        "ambient_occlusion_scale": "15",
        "ambient_occlusion_smooth": "15",
        "ray_trace_gain": "0.1",
        "ambient": "0.9",
        "direct": "0.2",
        "ray_trace_mode": "0",
    },
    "cartoon": {
        "cartoon_oval_length": "1.5",
        "cartoon_oval_width": "0.5",
        "cartoon_rect_length": "1.5",
        "cartoon_rect_width": "0.5",
        "cartoon_loop_radius": "0.3",
        "ray_trace_mode": "1",
        "ray_trace_color": "black",
        "opaque_background": "off",
    },
}


# in groups of three
DEFAULT_PROTEIN_COLORS = (0.8, 0.8, 0.6, 0.8, 0.6, 0.8, 0.6, 0.8, 0.8)
DEFAULT_HETATM_COLORS = (0.15, 0.7, 0.9, 0.9, 0.75, 0.15, 0.9, 0.15, 0.75)


def apply_render_style(render_style: str) -> None:
    """Apply render styles from a dict.
    Everything is global, because pymol.

    I cannot just use hasattr to tell what is cmd.set vs an attribute,
    because some attributes are both cmd attributes and cmd.set attributes
    e.g., valence is both an attribute of cmd and can be
    set with cmd.set and seems to have different meanings
    """
    from pymol import cmd

    if render_style in RENDER_OPTIONS:
        render_style_dict = RENDER_OPTIONS[render_style]
    else:
        render_style_dict = json.loads(render_style)

    for k, v in render_style_dict.items():
        if k == "bg_color":
            cmd.bg_color(v)
        else:
            cmd.set(k, v)


def get_orientation_for_ligand(
    pdb_file: str, ligand_id_or_chain: Union[str, tuple[str, str]]
) -> tuple[list, float]:
    import numpy as np
    from prody import calcCenter, parsePDB

    structure = parsePDB(pdb_file)
    if isinstance(ligand_id_or_chain, tuple):
        ligand_center = calcCenter(
            structure.select(
                f"resname {ligand_id_or_chain[0]} and chain {ligand_id_or_chain[1]}"
            )
        )
        nonligand_center = calcCenter(
            structure.select(
                f"not resname {ligand_id_or_chain[0]} and chain {ligand_id_or_chain[1]}"
            )
        )
    else:
        ligand_chain = structure.select(f"chain {ligand_id_or_chain}")
        ligand_resname = structure.select(f"resname {ligand_id_or_chain}")
        if ligand_chain is not None:
            ligand_center = calcCenter(structure.select(f"chain {ligand_id_or_chain}"))
            nonligand_center = calcCenter(
                structure.select(f"not chain {ligand_id_or_chain}")
            )
        elif ligand_resname is not None:
            # ideally arbitrarily pick one chain if there are multiple
            ligand_center = calcCenter(
                structure.select(f"resname {ligand_id_or_chain}")
            )
            nonligand_center = calcCenter(
                structure.select(f"not resname {ligand_id_or_chain}")
            )
        else:
            raise ValueError(
                f"Could not find ligand with id or chain {ligand_id_or_chain}"
            )

    # calculate rotation to orient ligand forward here
    forward_vector = ligand_center - nonligand_center
    forward_vector = forward_vector / np.linalg.norm(forward_vector)
    axis = np.cross(forward_vector, np.array([0, 0, 1]))
    angle = np.arccos(np.dot(forward_vector, np.array([0, 0, 1])))

    return [float(v) for v in axis], float(angle * 180 / np.pi)


@app.function(
    image=image,
    gpu=None,
    timeout=60 * 15,
)
def pdb2png(
    pdb_name: str,
    pdb_str: str,
    protein_rotates: list[tuple[float, float, float]] = None,
    protein_color: tuple[float, float, float] = None,
    protein_zoom: float = None,
    hetatm_color: tuple[float, float, float] = None,
    ligand_id: str = None,
    ligand_chain: str = None,
    ligand_zoom: float = None,
    ligand_color: str = "red",
    show_water: bool = False,
    render_style: str = "default",
    width: int = 1600,
    height: int = 1600,
) -> list:
    """
    Input is a pdb file.
    Output is a png file.
    """
    from pymol import cmd

    in_dir = "/tmp/in_pp"
    out_dir = "/tmp/out_pp"
    Path(in_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    in_pdb_file = Path(in_dir) / pdb_name
    open(in_pdb_file, "w").write(pdb_str)

    # --------------------------------------------------------------------------
    # Rotation
    #
    png_num = None
    png_num_str = ""

    for protein_rotate in protein_rotates or [None]:
        cmd.reinitialize()  # move here
        cmd.load(in_pdb_file)

        if protein_rotate is not None:
            cmd.rotate("x", protein_rotate[0])
            cmd.rotate("y", protein_rotate[1])
            cmd.rotate("z", protein_rotate[2])
            png_num = png_num + 1 if png_num is not None else 0
            png_num_str = f"_{png_num:04d}"

        elif ligand_id is not None or ligand_chain is not None:
            ligand_id_or_chain = (
                (ligand_id, ligand_chain)
                if ligand_id and ligand_chain
                else (ligand_id or ligand_chain)
            )
            _axis, _angle = get_orientation_for_ligand(
                str(in_pdb_file), ligand_id_or_chain
            )
            cmd.rotate(_axis, _angle)

        else:
            cmd.orient()

        # --------------------------------------------------------------------------
        # Colors
        #
        if protein_color is None:
            protein_color = DEFAULT_PROTEIN_COLORS

        if isinstance(protein_color, tuple):
            n = 0
            for chain in cmd.get_chains():
                cmd.set_color("protein_color", protein_color[n : n + 3])
                cmd.color("protein_color", f"chain {chain} and not hetatm")
                n = (n + 3) % len(protein_color)
        else:
            # color is a string like "red"
            cmd.color(protein_color, "not hetatm")

        # Color proteins and hetatms
        for hp_id, hp_color, hp_sel in [
            ("protein", protein_color, "not hetatm"),
            ("hetatm", hetatm_color, "hetatm"),
        ]:
            if hp_color is not None:
                if isinstance(hp_color, tuple):
                    n = 0
                    for chain in cmd.get_chains():
                        cmd.select(
                            f"sel_{hp_id}_{chain}", f"chain {chain} and {hp_sel}"
                        )
                        if cmd.count_atoms(f"sel_{hp_id}_{chain}") > 0:
                            cmd.set_color(f"{hp_id}_color_{chain}", hp_color[n : n + 3])
                            cmd.color(f"{hp_id}_color_{chain}", f"sel_{hp_id}_{chain}")
                            n = (n + 3) % len(hp_color)
                else:
                    cmd.color(hp_color, hp_sel)

        if protein_zoom is not None:
            cmd.zoom("all", protein_zoom)

        # --------------------------------------------------------------------------
        # Ligand
        #
        if ligand_id is not None:
            and_chain = f"and chain {ligand_chain}" if ligand_chain else ""
            cmd.select("ligand", f"resn {ligand_id} {and_chain}")

            if ligand_zoom is not None:
                cmd.zoom("ligand", ligand_zoom)

            if ligand_color is None:
                ligand_color = DEFAULT_HETATM_COLORS

            if isinstance(ligand_color, tuple):
                cmd.set_color("ligand_color", ligand_color)
                cmd.color("ligand_color", "ligand")
            else:
                cmd.color(ligand_color, "ligand")

        if not show_water:
            cmd.select("HOH", "resn HOH")
            cmd.hide("everything", "HOH")

        # --------------------------------------------------------------------------
        # Render and save
        #
        apply_render_style(render_style)
        cmd.ray(width, height)

        out_png_path = (
            Path(out_dir) / f"{Path(pdb_name).with_suffix('')}{png_num_str}.png"
        )
        out_png_path.parent.mkdir(parents=True, exist_ok=True)
        cmd.save(out_png_path, in_pdb_file)

    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*")
        if Path(out_file).is_file()
    ]


def _parse_rotation_range(rotate_str):
    """convert arg string to list of tuples for animation
    e.g., "100-200,0,0,10" -> [(100,0,0), (110,0,0), ...]
    """
    *ranges, num_steps = rotate_str.split(",")
    steps = int(num_steps)

    # if there is no range given, then just double up the number
    start_end = [
        (float(r), float(r))
        if "-" not in r
        else (float(r.split("-")[0]), float(r.split("-")[1]))
        for r in ranges
    ]

    steps_sizes = [(end - start) / steps for start, end in start_end]

    return [
        tuple(start + (step * i) for (start, _), step in zip(start_end, steps_sizes))
        for i in range(steps)
    ]


@app.local_entrypoint()
def main(
    input_pdb,
    protein_rotate: str = None,
    protein_color: str = None,
    protein_zoom: float = None,
    hetatm_color: str = None,
    ligand_id: str = None,
    ligand_chain: str = None,
    ligand_zoom: float = None,
    ligand_color: str = "red",
    show_water: bool = False,
    render_style: str = "default",
    width: int = 1600,
    height: int = 1600,
    out_dir: str = "./out/pdb2png",
    run_name: str = None,
):
    from datetime import datetime

    if protein_rotate is not None and "-" in protein_rotate:
        protein_rotates = _parse_rotation_range(protein_rotate)
    elif protein_rotate is not None:
        protein_rotates = [tuple(map(float, protein_rotate.split(",")))][:3]
    else:
        protein_rotates = None

    if protein_color is not None and "," in protein_color:
        protein_color = tuple(map(float, protein_color.split(",")))

    if ligand_color is not None and "," in ligand_color:
        ligand_color = tuple(map(float, ligand_color.split(",")))

    pdb_str = open(input_pdb).read()

    outputs = pdb2png.remote(
        Path(input_pdb).name,
        pdb_str,
        protein_rotates,
        protein_color,
        protein_zoom,
        hetatm_color,
        ligand_id,
        ligand_chain,
        ligand_zoom,
        ligand_color,
        show_water,
        render_style,
        width,
        height,
    )

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]

    for out_file, out_content in outputs:
        output_path = Path(out_dir) / (run_name or today) / Path(out_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(output_path, "wb") as out:
                out.write(out_content)
