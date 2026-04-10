import argparse
import os

import pandas as pd
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils_SSL import Radar_Dataset_eval, parse_protocol
from evaluation import calculate_EER
from model import Model


def resolve_config_path(explicit_path=None):
    config_path = explicit_path or "config.yaml"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return config_path


def resolve_model_paths(config):
    model_paths = config.get("model_paths")

    if isinstance(model_paths, str):
        model_paths = [model_paths]
    elif model_paths is None:
        model_paths = []
    elif not isinstance(model_paths, list):
        raise TypeError("`model_paths` must be a list of checkpoint paths or a string.")

    model_paths = [path for path in model_paths if path]
    if not model_paths:
        raise ValueError("No model checkpoints were provided in `model_paths`.")

    missing_paths = [path for path in model_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            "Missing model checkpoint(s): {}".format(", ".join(missing_paths))
        )

    return model_paths


def load_checkpoint_state(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_models(model_paths, args, device):
    models = []

    for checkpoint_path in model_paths:
        model = Model(args, device, args.ssl_model).to(device)
        state_dict = load_checkpoint_state(checkpoint_path, device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models


def average_model_outputs(models, batch_x):
    batch_outputs = [model(batch_x) for model in models]
    reference_shape = batch_outputs[0].shape

    for model_index, batch_out in enumerate(batch_outputs[1:], start=1):
        if batch_out.shape != reference_shape:
            raise ValueError(
                "Incompatible output shapes in ensemble: "
                f"model 0 -> {reference_shape}, model {model_index} -> {batch_out.shape}"
            )

    return torch.stack(batch_outputs, dim=0).mean(dim=0)


def produce_evaluation(data_loader, models, device, save_path):
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    val_loss = 0.0
    num_total = 0.0
    fname_list = []
    key_list = []
    score_list = []

    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)

            averaged_output = average_model_outputs(models, batch_x)
            batch_score = averaged_output[:, 1].detach().cpu().numpy().ravel().tolist()

            fname_list.extend(utt_id)
            # key_list.extend(["bonafide" if y.item() == 1 else "spoof" for y in batch_y])
            score_list.extend(batch_score)

    if not fname_list:
        raise ValueError("Evaluation dataset is empty. No scores were generated.")

    if len(fname_list) != len(score_list):
        raise ValueError("Mismatch between utterance IDs and generated scores.")

    score_df = pd.DataFrame(
        {
            "utt_id": fname_list,
            "score": score_list,
        }
    )
    score_df.to_csv(save_path, sep=" ", index=False, header=False)

    val_loss /= num_total
    print(f"Scores saved to {save_path}")

    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Evaluate model ensemble")
    parser.add_argument("--ssl_model", type=str, default="wavlm_large", help="SSL feature model name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--config", type=str, default=None, help="Path to the evaluation config file")
    args = parser.parse_args()

    conf_file = resolve_config_path(args.config)
    with open(conf_file, "r") as y_file:
        config = yaml.safe_load(y_file)

    cuda_device = config.get("cuda_device", "cuda:0")

    data_config = config["data_config"]
    data_name = data_config["dataset"]
    dataset_path = data_config["data_dir"]
    protocols_path = data_config["protocol_path"]
    protocol_filename = data_config["protocol_filename"]
    eval_protocol = os.path.join(protocols_path, protocol_filename)
    protocol_delimiter = data_config["protocol_delimiter"]
    protocol_file_id_column = data_config["protocol_file_id_column"]
    protocol_label_column = data_config["protocol_label_column"]
    bonfide_label = data_config["bonfide_label"]

    out_dir = config["score_dir"]
    os.makedirs(out_dir, exist_ok=True)

    model_paths = resolve_model_paths(config)
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
    models = load_models(model_paths, args, device)

    file_eval = parse_protocol(
        eval_protocol,
        delimiter=protocol_delimiter,
        key_col=protocol_file_id_column,
        bonafide_label=bonfide_label,
        has_label=False,
    )

    eval_set = Radar_Dataset_eval(
        list_IDs=file_eval,
        base_dir=dataset_path,
        config=config,
    )

    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        num_workers=data_config.get("num_workers", 8),
        shuffle=False,
    )

    out_score_file = os.path.join(out_dir, f"{data_name}_{config.eval_output}.txt")
    produce_evaluation(eval_loader, models, device, out_score_file)

    eval_eer = calculate_EER(cm_scores_file=out_score_file)
    eval_balanced_acc = None

    print(f"Loaded {len(models)} model(s) from config: {model_paths}")
    print(f"Eval EER: {eval_eer:.2f}%, Eval Balanced Acc: {eval_balanced_acc}")


if __name__ == "__main__":
    main()
