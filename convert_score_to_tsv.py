import argparse
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a whitespace-delimited score file to TSV."
    )
    parser.add_argument(
        "--score_file",
        required=True,
        help="Path to the input score file.",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        help="Optional output TSV path. Defaults to the input path with a .tsv suffix.",
    )
    return parser.parse_args()


def resolve_output_path(score_file, output_file=None):
    if output_file:
        return Path(output_file)
    return score_file.with_suffix(".tsv")


def main():
    args = parse_args()

    score_file = Path(args.score_file)
    if not score_file.is_file():
        raise FileNotFoundError(f"Score file not found: {score_file}")

    output_file = resolve_output_path(score_file, args.output_file)
    score_df = pd.read_csv(score_file, sep=r"\s+", header=None, engine="python")
    score_df.to_csv(output_file, sep="\t", index=False, header=False)

    print(f"TSV score file saved to {output_file}")


if __name__ == "__main__":
    main()
