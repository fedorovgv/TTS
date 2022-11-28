import os
import subprocess


WORK_DIRECTORY = os.getcwd()

SETUP_COMMANDS = [
    "chmod +x setup.sh",
    "./setup.sh",
    "export PYTHONPATH='${PYTHONPATH}:" + f"{WORK_DIRECTORY}'"
]

for i, command in enumerate(SETUP_COMMANDS):
    try:
      subprocess.check_call(command, shell=True)
      print(f'{i} :: OK :: {command}')
    except BaseException as e:
      print(f'{i} :: ERROR :: Can t do {command}!')
