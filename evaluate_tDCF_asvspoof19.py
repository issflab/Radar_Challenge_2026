import os
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import yaml

from data_utils_SSL import _normalize_delim
from evaluation import calculate_EER


def load_yaml_config(config_path):
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def gen_score_file(protocol_file_path, score_file_path, config, out_path=None):
    data_config = config["data_config"]
    protocol_delimiter = _normalize_delim(data_config.get("protocol_delimiter"))
    protocol_file_id_column = data_config["protocol_file_id_column"]
    protocol_label_column = data_config["protocol_label_column"]

    protocol_df = pd.read_csv(
        protocol_file_path,
        sep=protocol_delimiter,
        header=None,
        engine="python",
    )

    required_columns = max(protocol_file_id_column, protocol_label_column)
    if protocol_df.shape[1] <= required_columns:
        raise ValueError(
            "Protocol file does not contain the configured file-id/label columns: "
            f"expected indices up to {required_columns}, found {protocol_df.shape[1]} columns."
        )

    protocol_df = protocol_df[[protocol_file_id_column, protocol_label_column]].copy()
    protocol_df.columns = ["AUDIO_FILE_NAME", "KEY"]
    protocol_df["AUDIO_FILE_NAME"] = protocol_df["AUDIO_FILE_NAME"].astype(str).str.split('.').str[0]

    # Read score file expected as: filename score
    scores_df = pd.read_csv(
        score_file_path,
        sep=r"\s+",
        names=["AUDIO_FILE_NAME", "Scores"],
        engine="python",
    )
    scores_df["AUDIO_FILE_NAME"] = scores_df["AUDIO_FILE_NAME"].astype(str).str.split('.').str[0]

    merged_df = pd.merge(protocol_df, scores_df, on='AUDIO_FILE_NAME', how='inner')
    score_df = merged_df[['AUDIO_FILE_NAME', 'KEY', 'Scores']]

    if out_path is None:
        score_dir = os.path.dirname(score_file_path)
        score_stem = os.path.splitext(os.path.basename(score_file_path))[0]
        out_path = os.path.join(score_dir, f"{score_stem}-labels.txt")

    score_df.to_csv(out_path, sep=" ", header=None, index=False)
    return out_path

def compute_equal_error_rate(cm_score_file):

    print(cm_score_file)

    # Load CM scores
    cm_data = np.genfromtxt(cm_score_file, dtype=str)
    cm_utt_id = cm_data[:, 0]

    cm_keys = cm_data[:, 1]
    cm_scores = cm_data[:, 2].astype(float)

    # cm_sources = cm_data[:, 1]
    # cm_keys = cm_data[:, 2]
    # cm_scores = cm_data[:, 3].astype(float)

    # print(cm_utt_id)
    # print(cm_keys)
    # print(cm_scores)

    # cm_data = pd.read_csv(cm_score_file)

    # print(cm_data)

    # cm_utt_id = cm_data['AUDIO_FILE_NAME'].to_list()
    # cm_keys = cm_data['KEY'].to_list()
    # cm_scores = cm_data['Scores'].to_list()

    # other_cm_scores = -cm_scores

    # Extract bona fide (real human) and spoof scores from the CM scores
    # y_scores = [1.5, 2.0, 3.6, 1.2]
    # y_true = [1, -1, 1, 1]
    # bona_cm = [1.5, 3.6, 1.2]
    # spoof_cm = [2.0]

    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # EERs of the standalone systems and fix ASV operating point to EER threshold
    eer_cm = em.compute_eer(bona_cm, spoof_cm)[0]

    # other_eer_cm = em.compute_eer(other_cm_scores[cm_scores > 0], other_cm_scores[cm_scores <= 0])[0]

    # print(other_eer_cm)

    print('\nCM SYSTEM')
    # print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(min(eer_cm, other_eer_cm) * 100))
    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer_cm * 100))

    # return min(eer_cm, other_eer_cm)
    return eer_cm



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluation script')

    parser.add_argument('--score_file_has_keys', action='store_true', help='if score file has keys')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML config file')
    parser.add_argument('--score_file_dir', type=str, default='Score_Files/', help='Score File directory')
    parser.add_argument('--protocol_filepath', type=str, default='ASVspoof2019.LA.cm.eval.trl.txt', help='Path to the protocol file')
    parser.add_argument('--score_filename', type=str, default='scores-lfcc-gmm-512-asvspoof19-LA.txt', help='Path to the score file')

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    # protocol_file_path = args.score_file_dir + args.protocol_filename
    # protocol_file_path = os.path.join(config.db_folder, 'protocols', config.protocol_filenames[0])
    protocol_file_path = args.protocol_filepath

    score_file_path = os.path.join(args.score_file_dir, args.score_filename)

    if args.score_file_has_keys:
        eer = calculate_EER(score_file_path)
    
    else:
        labeled_score_path = gen_score_file(protocol_file_path, score_file_path, config)
        eer = calculate_EER(labeled_score_path)

    print('   EER            = {:8.5f} % (Equal error rate for countermeasure)'.format(eer))

######### Run example ###########

# python3 evaluate_tDCF_asvspoof19.py --config config.yaml --score_file_dir Score_Files --protocol_filepath /path/to/protocol.txt --score_filename my_scores.txt