from Bio.Phylo.TreeConstruction import DistanceTreeConstructor
from Bio.Phylo.TreeConstruction import DistanceCalculator
from Bio.Phylo.NewickIO import Writer
import subprocess
import dendropy as dp
import os, glob, re
import numpy as np

constructor = DistanceTreeConstructor()
tree_builder = constructor.nj
from .utilities import *


def simulate_data_alisim_from_tree(
        msa_dir,
        tree_dir,
        res_dir,
        bin_dir,
        model_options=None,
        seed=None,
        n_sims=1000,
        evol_model="GTR+G4+F",
        # indel settings
        ins_rate=0.01,
        del_rate=0.01,
        mean_len=3,
        ali_size=1000,
        evol_model_tag=None,
):
    # test sim ali
    rg = np.random.default_rng(seed)
    all_files = np.sort(glob.glob(os.path.join(msa_dir, "*")))
    tree_files = np.sort(glob.glob(os.path.join(tree_dir, "*")))

    rnd_indx = rg.choice(range(len(all_files)), size=n_sims, replace=False)
    if model_options is None:
        sub_model = ""
    else:
        sub_model = "_".join(x for x in model_options)
    if evol_model_tag is None:
        evol_model_tag = re.sub(r"[{}+,]", "_", evol_model)

    # Replace any character inside the brackets [] with '_'
    os.makedirs(
        os.path.join(res_dir, "alisim_" + sub_model + evol_model_tag), exist_ok=True
    )

    # build NJ tree
    if seed is None:
        seed = np.random.randint(low=0, high=1000)
    sim_i = 0
    for sim_i in rnd_indx:
        try:
            ali_file = all_files[sim_i]
            f_name = os.path.basename(ali_file).split(".")[0] + "_sim.fa"
            print(os.path.basename(ali_file))
            t_file = os.path.join(
                tree_dir, f_name.split("_sim")[0] + ".rootree"
            )  # tree_files[sim_i]
            print(t_file)

            t = dp.Tree.get_from_path(t_file, schema="newick")
            taxon_names = [taxon.label for taxon in t.taxon_namespace]

            anc_sequence_indx = 0  # rg.choice(range(len(taxon_names)))

            cmd = [
                "cd %s; " % bin_dir,
                "./iqtree3",
                "--alisim",
                os.path.join(res_dir, "alisim_" + sub_model + evol_model_tag, f_name),
                "-m",
                evol_model,
                "-t",
                t_file,
                "--length",
                str(ali_size),
                "--seed",
                str(seed + sim_i),
            ]

            if model_options is not None:
                if "indel" in model_options:
                    cmd = cmd + [
                        "--indel",
                        "%s,%s" % (ins_rate, del_rate),
                        str(mean_len),
                    ]

                if "variable_indel" in model_options:
                    ins_rate = rg.uniform(low=0, high=0.02)
                    del_rate = ins_rate + 0
                    mean_len = rg.integers(low=1, high=5)

                    cmd = cmd + [
                        "--indel",
                        "%s,%s" % (ins_rate, del_rate),
                        str(mean_len),
                    ]

                if "anc_seq" in model_options:
                    cmd = cmd + [
                        "--root-seq",
                        "%s,%s"
                        % (
                            str(ali_file),
                            taxon_names[anc_sequence_indx].replace(" ", "_"),
                        ),
                    ]
                if "CODON" in "_".join(model_options):
                    indx = [
                        i
                        for i in range(len(model_options))
                        if "CODON" in model_options[i]
                    ][0]
                    cmd = cmd + ["-st", model_options[indx]]

            cmd = cmd + []
            cmd = " ".join(cmd)

            if "MIXTURE" in evol_model_tag:
                cc = cmd.replace("""-m MIX""", """-m 'MIX""")
                cc = cc.replace("""}+G -t""", """}+G+F' -t""")
                cc = cc.replace("""cd """, """#!/bin/bash \ncd """)
                bash_command = cc

            else:
                cc = cmd.replace("""cd """, """#!/bin/bash \ncd """)
                bash_command = cc

            script_path = f"run_iqtree.sh"

            # 2. Write the string to a file
            with open(script_path, "w") as f:
                f.write(bash_command)

            # 3. Make the script executable (equivalent to chmod +x)
            os.chmod(script_path, 0o755)

            # 4. Execute the script
            try:
                # We use check=True so Python raises an error if IQ-TREE fails
                result = subprocess.run(
                    ["./" + script_path], check=True, capture_output=True, text=True
                )
                print("Command Output:\n", result.stdout)
            except subprocess.CalledProcessError as e:
                print("Error executing script:\n", e.stderr)

            # else:
            #     print(cmd)
            #     subprocess.run(cmd, cwd=bin_dir, check=True)

        except:
            pass

        sim_i += 1
