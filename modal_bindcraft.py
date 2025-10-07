# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""Runs the BindCraft protein binder design pipeline on Modal.

Adapting:
https://colab.research.google.com/github/martinpacesa/BindCraft/blob/main/notebooks/BindCraft.ipynb

Approximate cost for 3 designs, PDL1.pdb only:
- A10G = $2, 1.5h
- A100 = $3, 1h
- H100 = $4, 40m
"""

import os
from pathlib import Path

from modal import App, Image

# It is harder to provision GPUs if you set the timeout too high
GPU = os.environ.get("GPU", "L40S")
TIMEOUT = int(os.environ.get("TIMEOUT", 300))
print(f"Using GPU {GPU}; TIMEOUT {TIMEOUT}")


def set_up_pyrosetta():
    """Installs PyRosetta using pyrosettacolabsetup.

    Args:
        None

    Returns:
        None
    """
    import pyrosettacolabsetup

    pyrosettacolabsetup.install_pyrosetta(
        serialization=True, cache_wheel_on_google_drive=False
    )


image = (
    Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "aria2", "ffmpeg")
    .pip_install("numpy<2.0")  # Pin NumPy FIRST before any dependencies
    .pip_install(
        "pdb-tools==2.4.8", "ffmpeg-python==0.2.0", "plotly==5.18.0", "kaleido==0.2.1"
    )
    .pip_install("git+https://github.com/sokrypton/ColabDesign.git")
    .pip_install("pyrosettacolabsetup")
    .run_commands(
        "git clone https://github.com/martinpacesa/BindCraft /root/bindcraft",
        "cd /root/bindcraft && git checkout c0a48d595d4976694aa979438712ac94c16620bb",
        "chmod +x /root/bindcraft/functions/dssp",
        "chmod +x /root/bindcraft/functions/DAlphaBall.gcc",
    )
    .run_commands(
        "ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign && mkdir /params"
    )
    .run_commands(
        "aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
        " && mkdir -p /root/bindcraft/params"
        " && tar -xf alphafold_params_2022-12-06.tar -C /root/bindcraft/params"
    )
    .run_function(set_up_pyrosetta)
    .pip_install(
        "numpy<2.0",  # Re-enforce after pyrosetta (which may upgrade it)
        "jax[cuda]<0.7.0",  # Pin to avoid 'wraps' removal in JAX 0.7.0
        "matplotlib==3.8.1",  # https://github.com/martinpacesa/BindCraft/issues/4
    )
)


app = App("bindcraft", image=image)


@app.function(image=image, gpu=GPU, timeout=TIMEOUT * 60)
def bindcraft(
    design_path,
    binder_name,
    pdb_str,
    chains,
    target_hotspot_residues,
    lengths,
    number_of_final_designs,
    design_protocol="Default",
    interface_protocol="AlphaFold2",
    template_protocol="Default",
    filter_option="Default",
    max_trajectories: int | None = None,
):
    """Executes the BindCraft pipeline to design protein binders against a target structure.

    Args:
        design_path (str): Path for design outputs within the container.
        binder_name (str): Name for the binder design project.
        pdb_str (str): PDB file content as a string.
        chains (str): Target chain(s) in the PDB.
        target_hotspot_residues (str): Hotspot residues on the target.
        lengths (list[int]): Range of lengths for the binder.
        number_of_final_designs (int): Desired number of final designs.
        design_protocol (str): Design protocol to use (e.g., "Default", "Beta-sheet").
        interface_protocol (str): Interface protocol (e.g., "AlphaFold2", "MPNN").
        template_protocol (str): Template protocol (e.g., "Default", "Masked").
        filter_option (str): Filter settings to apply (e.g., "Default", "Peptide").
        max_trajectories (int | None): Maximum number of design trajectories to run.

    Returns:
        list[tuple[Path, bytes]]: A list of tuples, where each tuple contains the relative output
                                  file path from `design_path` and its byte content.
    """
    import json
    import os
    import shutil
    import time
    from datetime import datetime

    import numpy as np
    import pandas as pd
    from bindcraft.functions import (
        binder_hallucination,
        calc_ss_percentage,
        calculate_averages,
        calculate_clash_score,
        check_accepted_designs,
        check_filters,
        check_jax_gpu,
        check_n_trajectories,
        clear_mem,
        copy_dict,
        create_dataframe,
        generate_dataframe_labels,
        generate_directories,
        generate_filter_pass_csv,
        insert_data,
        load_af2_models,
        load_helicity,
        load_json_settings,
        predict_binder_complex,
        mk_afdesign_model,
        mpnn_gen_sequence,
        perform_advanced_settings_check,
        pr,
        pr_relax,
        predict_binder_alone,
        save_fasta,
        score_interface,
        target_pdb_rmsd,
        unaligned_rmsd,
        validate_design_sequence,
    )

    starting_pdb = f"/tmp/bindcraft/{binder_name}.pdb"
    Path(starting_pdb).parent.mkdir(parents=True, exist_ok=True)
    open(starting_pdb, "w").write(pdb_str)

    settings = {
        "design_path": design_path,
        "binder_name": binder_name,
        "starting_pdb": starting_pdb,
        "chains": chains,
        "target_hotspot_residues": target_hotspot_residues,
        "lengths": lengths,
        "number_of_final_designs": number_of_final_designs,
    }

    target_settings_path = f"/root/bindcraft/settings_target/{binder_name}.json"

    with open(target_settings_path, "w") as f:
        json.dump(settings, f, indent=4)

    # Advanced settings
    if design_protocol == "Default":
        design_protocol_tag = "default_4stage_multimer"
    elif design_protocol == "Beta-sheet":
        design_protocol_tag = "betasheet_4stage_multimer"
    elif design_protocol == "Peptide":
        design_protocol_tag = "peptide_3stage_multimer"
    else:
        raise ValueError("Unsupported design protocol")

    if interface_protocol == "AlphaFold2":
        interface_protocol_tag = ""
    elif interface_protocol == "MPNN":
        interface_protocol_tag = "_mpnn"
    else:
        raise ValueError("Unsupported interface protocol")

    if template_protocol == "Default":
        template_protocol_tag = ""
    elif template_protocol == "Masked":
        template_protocol_tag = "_flexible"
    else:
        raise ValueError("Unsupported template protocol")

    advanced_settings_path = (
        "/root/bindcraft/settings_advanced/"
        + design_protocol_tag
        + interface_protocol_tag
        + template_protocol_tag
        + ".json"
    )

    # Filters
    if filter_option == "Default":
        filter_settings_path = "/root/bindcraft/settings_filters/default_filters.json"
    elif filter_option == "Peptide":
        filter_settings_path = "/root/bindcraft/settings_filters/peptide_filters.json"
    elif filter_option == "Relaxed":
        filter_settings_path = "/root/bindcraft/settings_filters/relaxed_filters.json"
    elif filter_option == "Peptide_Relaxed":
        filter_settings_path = (
            "/root/bindcraft/settings_filters/peptide_relaxed_filters.json"
        )
    elif filter_option == "None":
        filter_settings_path = "/root/bindcraft/settings_filters/no_filters.json"
    else:
        raise ValueError("Unsupported filter type")

    args = {
        "settings": target_settings_path,
        "filters": filter_settings_path,
        "advanced": advanced_settings_path,
    }

    # Check if JAX-capable GPU is available, otherwise exit
    check_jax_gpu()

    # perform checks of input setting files
    settings_path, filters_path, advanced_path = (
        args["settings"],
        args["filters"],
        args["advanced"],
    )

    ### load settings from JSON
    target_settings, advanced_settings, filters = load_json_settings(
        settings_path, filters_path, advanced_path
    )

    print("target_settings", target_settings)
    print("advanced_settings", advanced_settings)
    print("filters", filters)
    if max_trajectories is not None:
        advanced_settings["max_trajectories"] = max_trajectories

    settings_file = os.path.basename(settings_path).split(".")[0]
    filters_file = os.path.basename(filters_path).split(".")[0]
    advanced_file = os.path.basename(advanced_path).split(".")[0]

    ### load AF2 model settings
    design_models, prediction_models, multimer_validation = load_af2_models(
        advanced_settings["use_multimer_design"]
    )

    ### perform checks on advanced_settings
    bindcraft_folder = "/root/bindcraft/"
    advanced_settings = perform_advanced_settings_check(
        advanced_settings, bindcraft_folder
    )

    ### generate directories, design path names can be found within the function
    design_paths = generate_directories(target_settings["design_path"])

    ### generate dataframes
    trajectory_labels, design_labels, final_labels = generate_dataframe_labels()

    trajectory_csv = os.path.join(
        target_settings["design_path"], "trajectory_stats.csv"
    )
    mpnn_csv = os.path.join(target_settings["design_path"], "mpnn_design_stats.csv")
    final_csv = os.path.join(target_settings["design_path"], "final_design_stats.csv")
    failure_csv = os.path.join(target_settings["design_path"], "failure_csv.csv")

    create_dataframe(trajectory_csv, trajectory_labels)
    create_dataframe(mpnn_csv, design_labels)
    create_dataframe(final_csv, final_labels)
    generate_filter_pass_csv(failure_csv, args["filters"])

    currenttime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Loaded design functions and settings at: {currenttime}")

    pr.init(
        f'-ignore_unrecognized_res -ignore_zero_occupancy -mute all -holes:dalphaball {advanced_settings["dalphaball_path"]} -corrections::beta_nov16 true -relax:default_repeats 1'
    )

    ####################################
    ###################### BindCraft Run
    ####################################
    # initialise counters
    script_start_time = time.time()
    trajectory_n = 1
    accepted_designs = 0

    ### start design loop
    while True:
        ### check if we have the target number of binders
        final_designs_reached = check_accepted_designs(
            design_paths,
            mpnn_csv,
            final_labels,
            final_csv,
            advanced_settings,
            target_settings,
            design_labels,
        )

        if final_designs_reached:
            # stop design loop execution
            break

        ### check if we reached maximum allowed trajectories
        # set advanced_settings["max_trajectories"]
        max_trajectories_reached = check_n_trajectories(design_paths, advanced_settings)

        if max_trajectories_reached:
            break

        ### Initialise design
        # measure time to generate design
        trajectory_start_time = time.time()

        # generate random seed to vary designs
        seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

        # sample binder design length randomly from defined distribution
        samples = np.arange(
            min(target_settings["lengths"]), max(target_settings["lengths"]) + 1
        )
        length = np.random.choice(samples)

        # load desired helicity value to sample different secondary structure contents
        helicity_value = load_helicity(advanced_settings)

        # generate design name and check if same trajectory was already run
        design_name = (
            target_settings["binder_name"] + "_l" + str(length) + "_s" + str(seed)
        )
        trajectory_dirs = [
            "Trajectory",
            "Trajectory/Relaxed",
            "Trajectory/LowConfidence",
            "Trajectory/Clashing",
        ]
        trajectory_exists = any(
            os.path.exists(
                os.path.join(design_paths[trajectory_dir], design_name + ".pdb")
            )
            for trajectory_dir in trajectory_dirs
        )

        if not trajectory_exists:
            print("Starting trajectory: " + design_name)

            ### Begin binder hallucination
            trajectory = binder_hallucination(
                design_name,
                target_settings["starting_pdb"],
                target_settings["chains"],
                target_settings["target_hotspot_residues"],
                length,
                seed,
                helicity_value,
                design_models,
                advanced_settings,
                design_paths,
                failure_csv,
            )
            trajectory_metrics = copy_dict(
                trajectory.aux["log"]
            )  # contains plddt, ptm, i_ptm, pae, i_pae
            trajectory_pdb = os.path.join(
                design_paths["Trajectory"], design_name + ".pdb"
            )

            # round the metrics to two decimal places
            trajectory_metrics = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in trajectory_metrics.items()
            }

            # time trajectory
            trajectory_time = time.time() - trajectory_start_time
            trajectory_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(trajectory_time // 3600), int((trajectory_time % 3600) // 60), int(trajectory_time % 60))}"
            print("Starting trajectory took: " + trajectory_time_text)
            print("")

            # Proceed if there is no trajectory termination signal
            if trajectory_metrics["terminate"] == "":
                # Relax binder to calculate statistics
                trajectory_relaxed = os.path.join(
                    design_paths["Trajectory/Relaxed"], design_name + ".pdb"
                )
                pr_relax(trajectory_pdb, trajectory_relaxed)

                # define binder chain, placeholder in case multi-chain parsing in ColabDesign gets changed
                binder_chain = "B"

                # Calculate clashes before and after relaxation
                num_clashes_trajectory = calculate_clash_score(trajectory_pdb)
                num_clashes_relaxed = calculate_clash_score(trajectory_relaxed)

                # secondary structure content of starting trajectory binder and interface
                (
                    trajectory_alpha,
                    trajectory_beta,
                    trajectory_loops,
                    trajectory_alpha_interface,
                    trajectory_beta_interface,
                    trajectory_loops_interface,
                    trajectory_i_plddt,
                    trajectory_ss_plddt,
                ) = calc_ss_percentage(trajectory_pdb, advanced_settings, binder_chain)

                # analyze interface scores for relaxed af2 trajectory
                (
                    trajectory_interface_scores,
                    trajectory_interface_AA,
                    trajectory_interface_residues,
                ) = score_interface(trajectory_relaxed, binder_chain)

                # starting binder sequence
                trajectory_sequence = trajectory.get_seq(get_best=True)[0]

                # analyze sequence
                traj_seq_notes = validate_design_sequence(
                    trajectory_sequence, num_clashes_relaxed, advanced_settings
                )

                # target structure RMSD compared to input PDB
                trajectory_target_rmsd = unaligned_rmsd(
                    target_settings["starting_pdb"],
                    trajectory_pdb,
                    target_settings["chains"],
                    "A",
                )

                # save trajectory statistics into CSV
                trajectory_data = [
                    design_name,
                    advanced_settings["design_algorithm"],
                    length,
                    seed,
                    helicity_value,
                    target_settings["target_hotspot_residues"],
                    trajectory_sequence,
                    trajectory_interface_residues,
                    trajectory_metrics["plddt"],
                    trajectory_metrics["ptm"],
                    trajectory_metrics["i_ptm"],
                    trajectory_metrics["pae"],
                    trajectory_metrics["i_pae"],
                    trajectory_i_plddt,
                    trajectory_ss_plddt,
                    num_clashes_trajectory,
                    num_clashes_relaxed,
                    trajectory_interface_scores["binder_score"],
                    trajectory_interface_scores["surface_hydrophobicity"],
                    trajectory_interface_scores["interface_sc"],
                    trajectory_interface_scores["interface_packstat"],
                    trajectory_interface_scores["interface_dG"],
                    trajectory_interface_scores["interface_dSASA"],
                    trajectory_interface_scores["interface_dG_SASA_ratio"],
                    trajectory_interface_scores["interface_fraction"],
                    trajectory_interface_scores["interface_hydrophobicity"],
                    trajectory_interface_scores["interface_nres"],
                    trajectory_interface_scores["interface_interface_hbonds"],
                    trajectory_interface_scores["interface_hbond_percentage"],
                    trajectory_interface_scores["interface_delta_unsat_hbonds"],
                    trajectory_interface_scores[
                        "interface_delta_unsat_hbonds_percentage"
                    ],
                    trajectory_alpha_interface,
                    trajectory_beta_interface,
                    trajectory_loops_interface,
                    trajectory_alpha,
                    trajectory_beta,
                    trajectory_loops,
                    trajectory_interface_AA,
                    trajectory_target_rmsd,
                    trajectory_time_text,
                    traj_seq_notes,
                    settings_file,
                    filters_file,
                    advanced_file,
                ]
                insert_data(trajectory_csv, trajectory_data)

                if advanced_settings["enable_mpnn"]:
                    # initialise MPNN counters
                    mpnn_n = 1
                    accepted_mpnn = 0
                    mpnn_dict = {}
                    design_start_time = time.time()

                    ### MPNN redesign of starting binder
                    mpnn_trajectories = mpnn_gen_sequence(
                        trajectory_pdb,
                        binder_chain,
                        trajectory_interface_residues,
                        advanced_settings,
                    )
                    existing_mpnn_sequences = set(
                        pd.read_csv(mpnn_csv, usecols=["Sequence"])["Sequence"].values
                    )

                    # create set of MPNN sequences with allowed amino acid composition
                    restricted_AAs = (
                        set(
                            aa.strip().upper()
                            for aa in advanced_settings["omit_AAs"].split(",")
                        )
                        if advanced_settings["force_reject_AA"]
                        else set()
                    )

                    mpnn_sequences = sorted(
                        {
                            mpnn_trajectories["seq"][n][-length:]: {
                                "seq": mpnn_trajectories["seq"][n][-length:],
                                "score": mpnn_trajectories["score"][n],
                                "seqid": mpnn_trajectories["seqid"][n],
                            }
                            for n in range(advanced_settings["num_seqs"])
                            if (
                                not restricted_AAs
                                or not any(
                                    aa in mpnn_trajectories["seq"][n][-length:].upper()
                                    for aa in restricted_AAs
                                )
                            )
                            and mpnn_trajectories["seq"][n][-length:]
                            not in existing_mpnn_sequences
                        }.values(),
                        key=lambda x: x["score"],
                    )

                    del existing_mpnn_sequences

                    # check whether any sequences are left after amino acid rejection and duplication check, and if yes proceed with prediction
                    if mpnn_sequences:
                        # add optimisation for increasing recycles if trajectory is beta sheeted
                        if (
                            advanced_settings["optimise_beta"]
                            and float(trajectory_beta) > 15
                        ):
                            advanced_settings[
                                "num_recycles_validation"
                            ] = advanced_settings["optimise_beta_recycles_valid"]

                        ### Compile prediction models once for faster prediction of MPNN sequences
                        clear_mem()
                        # compile complex prediction model
                        complex_prediction_model = mk_afdesign_model(
                            protocol="binder",
                            num_recycles=advanced_settings["num_recycles_validation"],
                            data_dir=advanced_settings["af_params_dir"],
                            use_multimer=multimer_validation,
                        )
                        complex_prediction_model.prep_inputs(
                            pdb_filename=target_settings["starting_pdb"],
                            chain=target_settings["chains"],
                            binder_len=length,
                            rm_target_seq=advanced_settings["rm_template_seq_predict"],
                            rm_target_sc=advanced_settings["rm_template_sc_predict"],
                        )

                        # compile binder monomer prediction model
                        binder_prediction_model = mk_afdesign_model(
                            protocol="hallucination",
                            use_templates=False,
                            initial_guess=False,
                            use_initial_atom_pos=False,
                            num_recycles=advanced_settings["num_recycles_validation"],
                            data_dir=advanced_settings["af_params_dir"],
                            use_multimer=multimer_validation,
                        )
                        binder_prediction_model.prep_inputs(length=length)

                        # iterate over designed sequences
                        for mpnn_sequence in mpnn_sequences:
                            mpnn_time = time.time()

                            # generate mpnn design name numbering
                            mpnn_design_name = design_name + "_mpnn" + str(mpnn_n)
                            mpnn_score = round(mpnn_sequence["score"], 2)
                            mpnn_seqid = round(mpnn_sequence["seqid"], 2)

                            # add design to dictionary
                            mpnn_dict[mpnn_design_name] = {
                                "seq": mpnn_sequence["seq"],
                                "score": mpnn_score,
                                "seqid": mpnn_seqid,
                            }

                            # save fasta sequence
                            if advanced_settings["save_mpnn_fasta"] is True:
                                save_fasta(
                                    mpnn_design_name, mpnn_sequence["seq"], design_paths
                                )

                            ### Predict mpnn redesigned binder complex using masked templates
                            (
                                mpnn_complex_statistics,
                                pass_af2_filters,
                            ) = predict_binder_complex(
                                complex_prediction_model,
                                mpnn_sequence["seq"],
                                mpnn_design_name,
                                target_settings["starting_pdb"],
                                target_settings["chains"],
                                length,
                                trajectory_pdb,
                                prediction_models,
                                advanced_settings,
                                filters,
                                design_paths,
                                failure_csv,
                            )

                            # if AF2 filters are not passed then skip the scoring
                            if not pass_af2_filters:
                                print(
                                    f"Base AF2 filters not passed for {mpnn_design_name}, skipping interface scoring"
                                )
                                mpnn_n += 1
                                continue

                            # calculate statistics for each model individually
                            for model_num in prediction_models:
                                mpnn_design_pdb = os.path.join(
                                    design_paths["MPNN"],
                                    f"{mpnn_design_name}_model{model_num+1}.pdb",
                                )
                                mpnn_design_relaxed = os.path.join(
                                    design_paths["MPNN/Relaxed"],
                                    f"{mpnn_design_name}_model{model_num+1}.pdb",
                                )

                                if os.path.exists(mpnn_design_pdb):
                                    # Calculate clashes before and after relaxation
                                    num_clashes_mpnn = calculate_clash_score(
                                        mpnn_design_pdb
                                    )
                                    num_clashes_mpnn_relaxed = calculate_clash_score(
                                        mpnn_design_relaxed
                                    )

                                    # analyze interface scores for relaxed af2 trajectory
                                    (
                                        mpnn_interface_scores,
                                        mpnn_interface_AA,
                                        mpnn_interface_residues,
                                    ) = score_interface(
                                        mpnn_design_relaxed, binder_chain
                                    )

                                    # secondary structure content of starting trajectory binder
                                    (
                                        mpnn_alpha,
                                        mpnn_beta,
                                        mpnn_loops,
                                        mpnn_alpha_interface,
                                        mpnn_beta_interface,
                                        mpnn_loops_interface,
                                        mpnn_i_plddt,
                                        mpnn_ss_plddt,
                                    ) = calc_ss_percentage(
                                        mpnn_design_pdb, advanced_settings, binder_chain
                                    )

                                    # unaligned RMSD calculate to determine if binder is in the designed binding site
                                    rmsd_site = unaligned_rmsd(
                                        trajectory_pdb,
                                        mpnn_design_pdb,
                                        binder_chain,
                                        binder_chain,
                                    )

                                    # calculate RMSD of target compared to input PDB
                                    target_rmsd = target_pdb_rmsd(
                                        mpnn_design_pdb,
                                        target_settings["starting_pdb"],
                                        target_settings["chains"],
                                    )

                                    # add the additional statistics to the mpnn_complex_statistics dictionary
                                    mpnn_complex_statistics[model_num + 1].update(
                                        {
                                            "i_pLDDT": mpnn_i_plddt,
                                            "ss_pLDDT": mpnn_ss_plddt,
                                            "Unrelaxed_Clashes": num_clashes_mpnn,
                                            "Relaxed_Clashes": num_clashes_mpnn_relaxed,
                                            "Binder_Energy_Score": mpnn_interface_scores[
                                                "binder_score"
                                            ],
                                            "Surface_Hydrophobicity": mpnn_interface_scores[
                                                "surface_hydrophobicity"
                                            ],
                                            "ShapeComplementarity": mpnn_interface_scores[
                                                "interface_sc"
                                            ],
                                            "PackStat": mpnn_interface_scores[
                                                "interface_packstat"
                                            ],
                                            "dG": mpnn_interface_scores["interface_dG"],
                                            "dSASA": mpnn_interface_scores[
                                                "interface_dSASA"
                                            ],
                                            "dG/dSASA": mpnn_interface_scores[
                                                "interface_dG_SASA_ratio"
                                            ],
                                            "Interface_SASA_%": mpnn_interface_scores[
                                                "interface_fraction"
                                            ],
                                            "Interface_Hydrophobicity": mpnn_interface_scores[
                                                "interface_hydrophobicity"
                                            ],
                                            "n_InterfaceResidues": mpnn_interface_scores[
                                                "interface_nres"
                                            ],
                                            "n_InterfaceHbonds": mpnn_interface_scores[
                                                "interface_interface_hbonds"
                                            ],
                                            "InterfaceHbondsPercentage": mpnn_interface_scores[
                                                "interface_hbond_percentage"
                                            ],
                                            "n_InterfaceUnsatHbonds": mpnn_interface_scores[
                                                "interface_delta_unsat_hbonds"
                                            ],
                                            "InterfaceUnsatHbondsPercentage": mpnn_interface_scores[
                                                "interface_delta_unsat_hbonds_percentage"
                                            ],
                                            "InterfaceAAs": mpnn_interface_AA,
                                            "Interface_Helix%": mpnn_alpha_interface,
                                            "Interface_BetaSheet%": mpnn_beta_interface,
                                            "Interface_Loop%": mpnn_loops_interface,
                                            "Binder_Helix%": mpnn_alpha,
                                            "Binder_BetaSheet%": mpnn_beta,
                                            "Binder_Loop%": mpnn_loops,
                                            "Hotspot_RMSD": rmsd_site,
                                            "Target_RMSD": target_rmsd,
                                        }
                                    )

                                    # save space by removing unrelaxed predicted mpnn complex pdb?
                                    if advanced_settings["remove_unrelaxed_complex"]:
                                        os.remove(mpnn_design_pdb)

                            # calculate complex averages
                            mpnn_complex_averages = calculate_averages(
                                mpnn_complex_statistics, handle_aa=True
                            )

                            ### Predict binder alone in single sequence mode
                            binder_statistics = predict_binder_alone(
                                binder_prediction_model,
                                mpnn_sequence["seq"],
                                mpnn_design_name,
                                length,
                                trajectory_pdb,
                                binder_chain,
                                prediction_models,
                                advanced_settings,
                                design_paths,
                            )

                            # extract RMSDs of binder to the original trajectory
                            for model_num in prediction_models:
                                mpnn_binder_pdb = os.path.join(
                                    design_paths["MPNN/Binder"],
                                    f"{mpnn_design_name}_model{model_num+1}.pdb",
                                )

                                if os.path.exists(mpnn_binder_pdb):
                                    rmsd_binder = unaligned_rmsd(
                                        trajectory_pdb,
                                        mpnn_binder_pdb,
                                        binder_chain,
                                        "A",
                                    )

                                # append to statistics
                                binder_statistics[model_num + 1].update(
                                    {"Binder_RMSD": rmsd_binder}
                                )

                                # save space by removing binder monomer models?
                                if advanced_settings["remove_binder_monomer"]:
                                    os.remove(mpnn_binder_pdb)

                            # calculate binder averages
                            binder_averages = calculate_averages(binder_statistics)

                            # analyze sequence to make sure there are no cysteins and it contains residues that absorb UV for detection
                            seq_notes = validate_design_sequence(
                                mpnn_sequence["seq"],
                                mpnn_complex_averages.get("Relaxed_Clashes", None),
                                advanced_settings,
                            )

                            # measure time to generate design
                            mpnn_end_time = time.time() - mpnn_time
                            elapsed_mpnn_text = f"{'%d hours, %d minutes, %d seconds' % (int(mpnn_end_time // 3600), int((mpnn_end_time % 3600) // 60), int(mpnn_end_time % 60))}"

                            # Insert statistics about MPNN design into CSV, will return None if corresponding model does note exist
                            model_numbers = range(1, 6)
                            statistics_labels = [
                                "pLDDT",
                                "pTM",
                                "i_pTM",
                                "pAE",
                                "i_pAE",
                                "i_pLDDT",
                                "ss_pLDDT",
                                "Unrelaxed_Clashes",
                                "Relaxed_Clashes",
                                "Binder_Energy_Score",
                                "Surface_Hydrophobicity",
                                "ShapeComplementarity",
                                "PackStat",
                                "dG",
                                "dSASA",
                                "dG/dSASA",
                                "Interface_SASA_%",
                                "Interface_Hydrophobicity",
                                "n_InterfaceResidues",
                                "n_InterfaceHbonds",
                                "InterfaceHbondsPercentage",
                                "n_InterfaceUnsatHbonds",
                                "InterfaceUnsatHbondsPercentage",
                                "Interface_Helix%",
                                "Interface_BetaSheet%",
                                "Interface_Loop%",
                                "Binder_Helix%",
                                "Binder_BetaSheet%",
                                "Binder_Loop%",
                                "InterfaceAAs",
                                "Hotspot_RMSD",
                                "Target_RMSD",
                            ]

                            # Initialize mpnn_data with the non-statistical data
                            mpnn_data = [
                                mpnn_design_name,
                                advanced_settings["design_algorithm"],
                                length,
                                seed,
                                helicity_value,
                                target_settings["target_hotspot_residues"],
                                mpnn_sequence["seq"],
                                mpnn_interface_residues,
                                mpnn_score,
                                mpnn_seqid,
                            ]

                            # Add the statistical data for mpnn_complex
                            for label in statistics_labels:
                                mpnn_data.append(mpnn_complex_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(
                                        mpnn_complex_statistics.get(model, {}).get(
                                            label, None
                                        )
                                    )

                            # Add the statistical data for binder
                            for label in [
                                "pLDDT",
                                "pTM",
                                "pAE",
                                "Binder_RMSD",
                            ]:  # These are the labels for binder alone
                                mpnn_data.append(binder_averages.get(label, None))
                                for model in model_numbers:
                                    mpnn_data.append(
                                        binder_statistics.get(model, {}).get(
                                            label, None
                                        )
                                    )

                            # Add the remaining non-statistical data
                            mpnn_data.extend(
                                [
                                    elapsed_mpnn_text,
                                    seq_notes,
                                    settings_file,
                                    filters_file,
                                    advanced_file,
                                ]
                            )

                            # insert data into csv
                            insert_data(mpnn_csv, mpnn_data)

                            # find best model number by pLDDT
                            plddt_values = {
                                i: mpnn_data[i]
                                for i in range(11, 15)
                                if mpnn_data[i] is not None
                            }

                            # Find the key with the highest value
                            highest_plddt_key = int(
                                max(plddt_values, key=plddt_values.get)
                            )

                            # Output the number part of the key
                            best_model_number = highest_plddt_key - 10
                            best_model_pdb = os.path.join(
                                design_paths["MPNN/Relaxed"],
                                f"{mpnn_design_name}_model{best_model_number}.pdb",
                            )

                            # run design data against filter thresholds
                            filter_conditions = check_filters(
                                mpnn_data, design_labels, filters
                            )
                            if filter_conditions is True:
                                print(mpnn_design_name + " passed all filters")
                                accepted_mpnn += 1
                                accepted_designs += 1

                                # copy designs to accepted folder
                                shutil.copy(best_model_pdb, design_paths["Accepted"])

                                # insert data into final csv
                                final_data = [""] + mpnn_data
                                insert_data(final_csv, final_data)

                                # copy animation from accepted trajectory
                                if advanced_settings["save_design_animations"]:
                                    accepted_animation = os.path.join(
                                        design_paths["Accepted/Animation"],
                                        f"{design_name}.html",
                                    )
                                    if not os.path.exists(accepted_animation):
                                        shutil.copy(
                                            os.path.join(
                                                design_paths["Trajectory/Animation"],
                                                f"{design_name}.html",
                                            ),
                                            accepted_animation,
                                        )

                                # copy plots of accepted trajectory
                                plot_files = os.listdir(
                                    design_paths["Trajectory/Plots"]
                                )
                                plots_to_copy = [
                                    f
                                    for f in plot_files
                                    if f.startswith(design_name) and f.endswith(".png")
                                ]
                                for accepted_plot in plots_to_copy:
                                    source_plot = os.path.join(
                                        design_paths["Trajectory/Plots"], accepted_plot
                                    )
                                    target_plot = os.path.join(
                                        design_paths["Accepted/Plots"], accepted_plot
                                    )
                                    if not os.path.exists(target_plot):
                                        shutil.copy(source_plot, target_plot)

                            else:
                                print(f"Unmet filter conditions for {mpnn_design_name}")
                                failure_df = pd.read_csv(failure_csv)
                                special_prefixes = (
                                    "Average_",
                                    "1_",
                                    "2_",
                                    "3_",
                                    "4_",
                                    "5_",
                                )
                                incremented_columns = set()

                                for column in filter_conditions:
                                    base_column = column
                                    for prefix in special_prefixes:
                                        if column.startswith(prefix):
                                            base_column = column.split("_", 1)[1]

                                    if base_column not in incremented_columns:
                                        failure_df[base_column] = (
                                            failure_df[base_column] + 1
                                        )
                                        incremented_columns.add(base_column)

                                failure_df.to_csv(failure_csv, index=False)
                                shutil.copy(best_model_pdb, design_paths["Rejected"])

                            # increase MPNN design number
                            mpnn_n += 1

                            # if enough mpnn sequences of the same trajectory pass filters then stop
                            if accepted_mpnn >= advanced_settings["max_mpnn_sequences"]:
                                break

                        if accepted_mpnn >= 1:
                            print(
                                "Found "
                                + str(accepted_mpnn)
                                + " MPNN designs passing filters"
                            )
                        else:
                            print("No accepted MPNN designs found for this trajectory.")

                    else:
                        print(
                            "Duplicate MPNN designs sampled with different trajectory, skipping current trajectory optimisation"
                        )

                    # save space by removing unrelaxed design trajectory PDB
                    if advanced_settings["remove_unrelaxed_trajectory"]:
                        os.remove(trajectory_pdb)

                    # measure time it took to generate designs for one trajectory
                    design_time = time.time() - design_start_time
                    design_time_text = f"{'%d hours, %d minutes, %d seconds' % (int(design_time // 3600), int((design_time % 3600) // 60), int(design_time % 60))}"
                    print(
                        "Design and validation of trajectory "
                        + design_name
                        + " took: "
                        + design_time_text
                    )

                # analyse the rejection rate of trajectories to see if we need to readjust the design weights
                if (
                    trajectory_n >= advanced_settings["start_monitoring"]
                    and advanced_settings["enable_rejection_check"]
                ):
                    acceptance = accepted_designs / trajectory_n
                    if not acceptance >= advanced_settings["acceptance_rate"]:
                        print(
                            "The ratio of successful designs is lower than defined acceptance rate! Consider changing your design settings!"
                        )
                        print("Script execution stopping...")
                        break

            # increase trajectory number
            trajectory_n += 1

    ### Script finished
    elapsed_time = time.time() - script_start_time
    elapsed_text = f"{'%d hours, %d minutes, %d seconds' % (int(elapsed_time // 3600), int((elapsed_time % 3600) // 60), int(elapsed_time % 60))}"
    print(
        "Finished all designs. Script execution for "
        + str(trajectory_n)
        + " trajectories took: "
        + elapsed_text
    )

    # Consolidate & Rank Designs
    accepted_binders = [
        f for f in os.listdir(design_paths["Accepted"]) if f.endswith(".pdb")
    ]

    for f in os.listdir(design_paths["Accepted/Ranked"]):
        os.remove(os.path.join(design_paths["Accepted/Ranked"], str(f)))

    # load dataframe of designed binders
    design_df = pd.read_csv(mpnn_csv)
    design_df = design_df.sort_values("Average_i_pTM", ascending=False)

    # create final csv dataframe to copy matched rows, initialize with the column labels
    final_df = pd.DataFrame(columns=final_labels)

    # check the ranking of the designs and copy them with new ranked IDs to the folder
    rank = 1
    for _, row in design_df.iterrows():
        for binder in accepted_binders:
            target_settings["binder_name"], model = binder.rsplit("_model", 1)
            if target_settings["binder_name"] == row["Design"]:
                # rank and copy into ranked folder
                row_data = {
                    "Rank": rank,
                    **{label: row[label] for label in design_labels},
                }
                final_df = pd.concat(
                    [final_df, pd.DataFrame([row_data])], ignore_index=True
                )
                old_path = os.path.join(design_paths["Accepted"], binder)
                new_path = os.path.join(
                    design_paths["Accepted/Ranked"],
                    f"{rank}_{target_settings['binder_name']}_model{model.rsplit('.', 1)[0]}.pdb",
                )
                shutil.copyfile(old_path, new_path)

                rank += 1
                break

    # save the final_df to final_csv
    final_df.to_csv(final_csv, index=False)

    out_dir = design_path
    return [
        (out_file.relative_to(out_dir), open(out_file, "rb").read())
        for out_file in Path(out_dir).glob("**/*.*")
    ]


@app.local_entrypoint()
def main(
    input_pdb: str,
    target_chains: str = "A",
    target_hotspot_residues: str = "",
    lengths: str = "50,130",
    number_of_final_designs: int = 1,
    max_trajectories: int | None = None,
    binder_name: str | None = None,
    out_dir: str = "./out/bindcraft",
    run_name: str | None = None,
):
    """Local entrypoint to run BindCraft binder design.

    Args:
        input_pdb (str): Path to the input PDB file.
        target_chains (str, optional): Target chain(s) in the PDB. Defaults to "A".
        target_hotspot_residues (str, optional): Hotspot residues on the target.
            For example "1,2-10" or chain specific "A1-10,B1-20" or entire chains "A".
            If left blank, an appropriate site will be selected by the pipeline.
            Defaults to "".
        lengths (str, optional): Comma-separated string defining the range of lengths for the binder
                                 (e.g., "50,130"). Defaults to "50,130".
        number_of_final_designs (int, optional): Desired number of final designs. Defaults to 1.
        max_trajectories (int | None, optional): Maximum number of design trajectories to run.
                                                 Defaults to None.
        binder_name (str | None, optional): Name for the binder design project. If None, it's derived
                                            from the input PDB filename. Defaults to None.
        out_dir (str, optional): Directory to save the output files. Defaults to "./out/bindcraft".
        run_name (str | None, optional): Optional name for the run, used to create a subdirectory
                                         in `out_dir`. If None, a timestamp-based name is used.
                                         Defaults to None.

    Returns:
        None
    """
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d%H%M")[2:]

    pdb_str = open(input_pdb).read()
    binder_name = binder_name or Path(input_pdb).stem
    design_path = f"/tmp/BindCraft/{binder_name}/"
    lengths_list = [int(i) for i in lengths.split(",")]

    outputs = bindcraft.remote(
        design_path=design_path,
        binder_name=binder_name,
        pdb_str=pdb_str,
        chains=target_chains,
        target_hotspot_residues=target_hotspot_residues,
        lengths=lengths_list,
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
    )

    for out_file, out_content in outputs:
        output_path = Path(out_dir) / (run_name or today) / out_file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as out:
            out.write(out_content)
