"""
Visualize a pdb file as a png.

TODO: center on the ligand, if there is one, and find the best orientation.
"""

import json
from pathlib import Path
from typing import Union

from modal import App, Image, Mount

FORCE_BUILD = False
LOCAL_IN = "./in/pdb2png"
LOCAL_OUT = "./out/pdb2png"
REMOTE_IN = "/in"

app = App()

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
    "muted": {
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
            structure.select(f"resname {ligand_id_or_chain[0]} and chain {ligand_id_or_chain[1]}")
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
            nonligand_center = calcCenter(structure.select(f"not chain {ligand_id_or_chain}"))
        elif ligand_resname is not None:
            # ideally arbitrarily pick one chain if there are multiple
            ligand_center = calcCenter(structure.select(f"resname {ligand_id_or_chain}"))
            nonligand_center = calcCenter(structure.select(f"not resname {ligand_id_or_chain}"))
        else:
            raise ValueError(f"Could not find ligand with id or chain {ligand_id_or_chain}")

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
    mounts=[Mount.from_local_dir(LOCAL_IN, remote_path=REMOTE_IN)],
)
def pdb2png(
    pdb_file: str,
    protein_rotate: Union[tuple[float, float, float], None] = None,
    protein_color: Union[tuple[float, float, float], str, None] = None,
    protein_zoom: Union[float, None] = None,
    hetatm_color: Union[tuple[float, float, float], str, None] = None,
    ligand_id: Union[str, None] = None,
    ligand_chain: Union[str, None] = None,
    ligand_zoom: float = None,
    ligand_color: Union[tuple[float, float, float], str] = "red",
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

    in_pdb_file = Path(REMOTE_IN) / Path(pdb_file).relative_to(LOCAL_IN)

    # reinitialize is important if you call the function multiple times!
    cmd.reinitialize()

    cmd.load(in_pdb_file)
    cmd.orient()  # maybe not needed

    #
    # Rotation
    #
    if protein_rotate is not None:
        cmd.rotate("x", protein_rotate[0])
        cmd.rotate("y", protein_rotate[1])
        cmd.rotate("z", protein_rotate[2])
    elif ligand_id is not None or ligand_chain is not None:
        ligand_id_or_chain = (
            (ligand_id, ligand_chain) if ligand_id and ligand_chain else (ligand_id or ligand_chain)
        )
        _axis, _angle = get_orientation_for_ligand(str(in_pdb_file), ligand_id_or_chain)
        cmd.rotate(_axis, _angle)

    #
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
                    cmd.select(f"sel_{hp_id}_{chain}", f"chain {chain} and {hp_sel}")
                    if cmd.count_atoms(f"sel_{hp_id}_{chain}") > 0:
                        cmd.set_color(f"{hp_id}_color_{chain}", hp_color[n : n + 3])
                        cmd.color(f"{hp_id}_color_{chain}", f"sel_{hp_id}_{chain}")
                        n = (n + 3) % len(hp_color)
            else:
                cmd.color(hp_color, hp_sel)

    if protein_zoom is not None:
        cmd.zoom("all", protein_zoom)

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

    #
    # Render
    #
    apply_render_style(render_style)

    cmd.ray(width, height)

    out_png_path = Path(LOCAL_OUT) / Path(pdb_file).relative_to(LOCAL_IN).with_suffix(".png")
    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    cmd.save(out_png_path, in_pdb_file)
    return [(out_png_path, open(out_png_path, "rb").read())]


@app.local_entrypoint()
def main(
    input_pdb,
    protein_rotate=None,
    protein_color=None,
    protein_zoom=None,
    hetatm_color=None,
    ligand_id=None,
    ligand_chain=None,
    ligand_zoom=None,
    ligand_color="red",
    show_water=False,
    render_style="default",
    width=1600,
    height=1600,
) -> list:
    if isinstance(protein_rotate, str):
        protein_rotate = tuple(map(float, protein_color.split(",")))
    if isinstance(protein_color, str) and "," in protein_color:
        protein_color = tuple(map(float, protein_color.split(",")))
    if isinstance(ligand_color, str) and "," in ligand_color:
        ligand_color = tuple(map(float, ligand_color.split(",")))

    outputs = pdb2png.remote(
        input_pdb,
        protein_rotate,
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

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
