import subprocess

commands = [
    "python -m spacy download en_core_web_md"
    # ,"python -m spacy link en_core_web_md en"
]

for command in commands:
    subprocess.run(command, shell=True)