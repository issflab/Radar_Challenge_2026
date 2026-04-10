import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a protocol file from files in a directory."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to the directory containing input files.",
    )
    parser.add_argument(
        "--protocol_name",
        required=True,
        help="Name of the protocol file to create inside input_dir.",
    )
    parser.add_argument(
        "--include-extension-in-id",
        action="store_true",
        help="Use the full filename, including extension, as file_id.",
    )
    parser.add_argument(
        "--extension",
        default=None,
        help="Only include files with this extension, for example .wav.",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Delimiter to place between file_id and file_path. Default is a space.",
    )
    return parser.parse_args()


def normalize_extension(extension):
    if extension is None:
        return None
    return extension if extension.startswith(".") else f".{extension}"


def collect_files(input_dir, extension=None):
    files = []
    for path in sorted(input_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        if extension is not None and path.suffix.lower() != extension.lower():
            continue
        files.append(path)
    return files


def build_protocol_lines(files, include_extension_in_id, delimiter):
    lines = []
    for file_path in files:
        file_id = file_path.name if include_extension_in_id else file_path.stem
        lines.append(f"{file_id}{delimiter}{file_path.resolve()}")
    return lines


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    extension = normalize_extension(args.extension)
    
    flac_dir = input_dir / 'flac'
    
    files = collect_files(flac_dir, extension=extension)
    if not files:
        raise ValueError(f"No valid files found in: {input_dir}")

    output_path = input_dir / args.protocol_name
    lines = build_protocol_lines(files, args.include_extension_in_id, args.delimiter)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Protocol saved to {output_path}")
    print(f"Wrote {len(lines)} entries")


if __name__ == "__main__":
    main()

############# Run it Like This #############
# python3 generate_protocol.py --input_dir /data/radar_challenge_data/RADAR2026-dev --protocol_name dev_protocol.txt --extension .flac
