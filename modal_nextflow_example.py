# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""
Basic example script running nextflow on modal.
Does not do anything except get the installation working.
Some of the installation steps are probably not necessary.
"""
from subprocess import run

from modal import App, Image

nextflow_script = '''#!/usr/bin/env nextflow
process sayHello {
    output:
        stdout
    """
    echo 'Hello World!'
    """
}

workflow {
    sayHello()
}
'''

image = (
    Image.debian_slim()
    .apt_install("git", "wget", "curl", "unzip", "zip", "procps")
    .pip_install("polars")
    #
    # Install nextflow
    #
    # no idea why sdkman is so difficult to install but it was a pain
    .run_commands(["rm /bin/sh && ln -s /bin/bash /bin/sh"])
    .run_commands('curl -s "https://get.sdkman.io" | bash')
    .env({"BASH_ENV": "/root/.sdkman/bin/sdkman-init.sh"})
    .run_commands("/bin/bash -c 'sdk install java 17.0.10-tem'")
    .run_commands("curl -s https://get.nextflow.io | bash")
    #
    # Install conda (taken from modal slack, not all steps necessary)
    #
    .run_commands(
        "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && "
        "chmod +x Miniconda3-latest-Linux-x86_64.sh && "
        "./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3 && "
        'export PATH="$HOME/miniconda3/bin:$PATH" && '
        'echo "export PATH=$HOME/miniconda3/bin:$PATH" >> /etc/environment && '
        "source $HOME/miniconda3/etc/profile.d/conda.sh && "
        "conda init && "
        "source ~/.bashrc"
    )
)


app = App("nextflow_example", image=image)


@app.function()
def run_nextflow():
    with open("hello-world.nf", "w") as out:
        out.write(nextflow_script)

    cmd = "/nextflow run hello-world.nf"
    run(cmd, shell=True, check=True)


@app.local_entrypoint()
def main():
    run_nextflow.remote()
