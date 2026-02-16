"""Split a dataset into its respective channels. This script acts on datasets in the format produced by the OMEConverter class.
- images
- masks (default produced by ideas)
- metadata
"""

import argparse
from pathlib import Path


def setup_dirs(output:Path):
    output.mkdir(parents=True, exist_ok=True)


def get_filenames(input:Path, channel:int) -> list[Path]:
    filenames: list[Path] = []

    for filename in input.rglob(f"*Ch{channel}*"):
        filenames.append(filename)

    return filenames

def copy_files(filenames:list[Path], output:Path):
    for filename in filenames:
        filename.copy_into(output)

def main(input_str:str, output_str:str, channel:int):
    # Convert strings to paths
    input_path = Path(input_str).joinpath("images")
    output_path = Path(output_str).joinpath("images")

    # Ensure that the target location exists
    setup_dirs(output_path)

    # Get all the filenames
    files_to_move = get_filenames(input_path, channel)

    # Copy the files to the target locations
    copy_files(files_to_move, output_path)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", '-i', type=str)
    parser.add_argument("--output-path", "-o", type=str)
    parser.add_argument("--channel", "-c", default=1, type=int)

    args = parser.parse_args()

    main(args.input_path, args.output_path, args.channel)
    