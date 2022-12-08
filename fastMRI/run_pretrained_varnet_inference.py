"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import fastmri
import fastmri.data.transforms as T
import numpy as np
import requests
import torch
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data import SliceDataset
from fastmri.models import VarNet
from tqdm import tqdm

VARNET_FOLDER = "https://dl.fbaipublicfiles.com/fastMRI/trained_models/varnet/"
MODEL_FNAMES = {
    "varnet_knee_mc": "knee_leaderboard_state_dict.pt",
    "varnet_brain_mc": "brain_leaderboard_state_dict.pt",
}


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def run_varnet_model(batch, model, device):
    masked_kspace, mask, _, fname, slice_num, _, crop_size = batch
    crop_size = crop_size[0]  # always have a batch size of 1 for varnet

    output = model(masked_kspace.to(device), mask.to(device)).cpu()

    # detect FLAIR 203
    if output.shape[-1] < crop_size[1]:
        crop_size = (output.shape[-1], output.shape[-1])

    output = T.center_crop(output, crop_size)[0]

    return output, int(slice_num[0]), fname[0]


def run_inference(challenge, state_dict_file, data_path, output_path, accelerations, center_fractions, mask_type, device):
    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    # download the state_dict if we don't have it
    if state_dict_file is None:
        if not Path(MODEL_FNAMES[challenge]).exists():
            url_root = VARNET_FOLDER
            download_model(url_root + MODEL_FNAMES[challenge], MODEL_FNAMES[challenge])

        state_dict_file = MODEL_FNAMES[challenge]

    model.load_state_dict(torch.load(state_dict_file))
    model = model.eval()

    mask = create_mask_for_mask_type(
        mask_type, center_fractions, accelerations
    )

    # data loader setup
    data_transform = T.VarNetDataTransform(mask_func=mask)
    dataset = SliceDataset(
        root=data_path, transform=data_transform, challenge="multicoil"
    )
    dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)

    # run the model
    start_time = time.perf_counter()
    outputs = defaultdict(list)
    model = model.to(device)

    for batch in tqdm(dataloader, desc="Running inference"):
        with torch.no_grad():
            output, slice_num, fname = run_varnet_model(batch, model, device)

        outputs[fname].append((slice_num, output))

    # save outputs
    for fname in outputs:
        outputs[fname] = np.stack([out for _, out in sorted(outputs[fname])])

    final_output_path = output_path / f"rec_acc_{accelerations[0]}"
    print('Saving to: ', final_output_path)
    fastmri.save_reconstructions(outputs, final_output_path)
    
    end_time = time.perf_counter()

    print(f"Elapsed time for {len(dataloader)} slices: {end_time-start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--challenge",
        default="varnet_brain_mc",
        choices=(
            "varnet_knee_mc",
            "varnet_brain_mc",
        ),
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Model to run",
    )
    parser.add_argument(
        "--state_dict_file",
        default=None,
        type=Path,
        help="Path to saved state_dict (will download if not provided)",
    )
    parser.add_argument(
        "--data_path",
        default='/mn/kadingir/vegardantun_000000/nobackup/fastMRI/multicoil_val_small2',
        type=Path,
        required=False,
        help="Path to subsampled data",
    )
    parser.add_argument(
        "--output_path",
        default='/mn/kadingir/vegardantun_000000/nobackup/fastMRI/multicoil_val_rec',
        type=Path,
        required=False,
        help="Path for saving reconstructions",
    )


    # data transform params
    parser.add_argument(
        "--mask_type",
        choices=("random", "equispaced"),
        default="equispaced",
        type=str,
        help="Type of k-space mask",
    )
    parser.add_argument(
        "--center_fractions",
        nargs="+",
        default=[0.04],
        type=float,
        help="Number of center lines to use in mask",
    )
    parser.add_argument(
        "--accelerations",
        nargs="+",
        default=[8],
        type=int,
        help="Acceleration rates to use for masks",
    )


    args = parser.parse_args()

    run_inference(
        args.challenge,
        args.state_dict_file,
        args.data_path,
        args.output_path,
        args.accelerations,
        args.center_fractions,
        args.mask_type,
        torch.device(args.device),
    )
