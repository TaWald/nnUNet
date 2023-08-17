from pathlib import Path
import os
import shutil
import sys
import time
from typing import Sequence
import nrrd
import numpy as np
import scipy
from tqdm import tqdm
import vtk
import SimpleITK as sitk
from scipy.ndimage import morphology
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
import torch
from torch.cuda import OutOfMemoryError
from skimage import morphology as sk_morphology

from totalsegmentator.python_api import totalsegmentator


def convert_types(nrrd_file: str, nifti_file: str):
    """Convert nrrd file to nifti file."""
    im = sitk.ReadImage(nrrd_file)
    sitk.WriteImage(im, nifti_file)


def convert(input_image: list[Path], output_path: Path, append_index_suffix=False):
    """Convert all nrrd files in input_images to nifti files in output_path."""
    output_path.mkdir(exist_ok=True, parents=True)
    for im in tqdm(input_image):
        im_name = im.name

        nifti_name = im_name[:-5] + (
            "_0000.nii.gz" if append_index_suffix else ".nii.gz"
        )
        if (output_path / nifti_name).exists():
            continue
        else:
            convert_types(str(im), str(output_path / nifti_name))


def total_segmentator_predict_dir(case_dir, output_dir):
    """Run total segmentator on all nifti files in case_dir and save the results in output_dir."""
    output_dir.mkdir(exist_ok=True, parents=True)
    for im in tqdm(case_dir.iterdir()):
        im_name = im.name
        if not im_name.endswith(".nii.gz"):
            raise ValueError(f"File {im} is not a nifti file.")
        out_name = im_name[:-7] + "_seg.nii.gz"
        if (output_dir / out_name).exists():
            print("Skipping total segmentator creation as file exists already!")
            continue
        else:
            print(f"Running total segmentator on {im_name}")
            totalsegmentator(
                str(im),
                str(output_dir / out_name),
                nr_thr_resamp=3,
                nr_thr_saving=6,
                ml=True,
                fast=False,
                force_split=False,
            )


def anatomical_score_dir(case_dir, output_dir):
    """Run total segmentator on all nifti files in case_dir and save the results in output_dir."""
    output_dir.mkdir(exist_ok=True, parents=True)
    for im in tqdm(case_dir.iterdir()):
        im_name = im.name
        if not im_name.endswith(".nii.gz"):
            raise ValueError(f"File {im} is not a nifti file.")
        out_name = im_name[:-7] + "_seg.nii.gz"
        if (output_dir / out_name).exists():
            print("Skipping total segmentator creation as file exists already!")
            continue
        else:
            totalsegmentator(str(im), str(output_dir / out_name), ml=True)


def filter_non_same_size_cases(case: list[Path], label: list[Path]):
    mismatched_cases = []
    acceptable_cases_im = []
    acceptable_cases_lbl = []
    for c, l in zip(case, label):
        c_header = nrrd.read_header(str(c))
        l_header = nrrd.read_header(str(l))
        if np.any(c_header["sizes"] != l_header["sizes"]):
            mismatched_cases.append((c, l, c_header, l_header))
        else:
            acceptable_cases_im.append(c)
            acceptable_cases_lbl.append(l)
    # for mc in sorted(mismatched_cases, key=lambda x: x[0]):
    #     print(f"Case mismatch: {mc[0]} {mc[1]} - Shapes: {mc[2]['sizes']} {mc[3]['sizes']} - Spacings: \n{mc[2]['space directions']} \n{mc[3]['space directions']}")
    return acceptable_cases_im, acceptable_cases_lbl


def get_ids_from_dir(dir: Path):
    ids = []
    for case in dir.iterdir():
        name = case.name
        if not name.endswith(".nrrd"):
            continue
        ids.append(name.split("-")[-2])
    return list(set(ids))


def get_train_ids_and_im_path_from_dir(dir: Path) -> dict[int, Path]:
    """Intended vor the LNQ directory that contains both image and segmentation (and that other garbage)"""
    ids: dict[int, Path] = {}
    for case in dir.iterdir():
        name = case.name
        if "ct" in name:
            ids[int(name.split("-")[-2])] = case
    return ids


def get_ids_and_path_from_dir(dir: Path) -> dict[int, Path]:
    """Intended vor the LNQ directory that contains only the created segmentations"""
    ids: dict[int, Path] = {}
    for case in dir.iterdir():
        name = case.name
        ids[int(name.split("-")[-2])] = case
    return ids


def get_im_and_label_from_id(id: str, dir: Path, is_val: bool) -> tuple[Path, Path]:
    im = dir / f"lnq2023-{'val' if is_val else 'train'}-{id}-ct.nrrd"
    label = dir / f"lnq2023-{'val' if is_val else 'train'}-{id}-seg.nrrd"
    return im, label


def find_mutually_exclusive_classes_with_total_segmentator(
    total_segmentator_dir: Path, groundtruth_dir: Path, out_path: Path
):
    """Loads the both niftis, then compares the masks of all classes to the groundtruth mask. Calculates if classes overlap and if so, how much percent of the groundtruth is covered by the class."""
    if out_path.exists():
        out = load_json(str(out_path))
        mean_res = out["mean_res"]
        non_zero_mean = out["non_zero_mean"]
        all_results = out["all_results"]
    else:
        all_total_segmentator_files = list(sorted(os.listdir(total_segmentator_dir)))
        all_groundtruth_files = list(sorted(os.listdir(groundtruth_dir)))

        for ts, gt in zip(all_total_segmentator_files, all_groundtruth_files):
            assert ts.split("-")[2] == gt.split("-")[2], (
                "Total segmentator and groundtruth files do not match!"
                + f"N_Totalsegmentator: {len(all_total_segmentator_files)}, N_GT: {len(all_groundtruth_files)}"
                + f"Case ids: TS: {ts}, GT: {gt}"
            )

        all_results = {}
        for i in range(105):
            all_results[i] = []

        for ts, gt in tqdm(zip(all_total_segmentator_files, all_groundtruth_files)):
            ts_im = sitk.ReadImage(total_segmentator_dir / ts)
            gt_im = sitk.ReadImage(groundtruth_dir / gt)

            ts_data = sitk.GetArrayFromImage(ts_im)
            lymphnode_data = sitk.GetArrayFromImage(gt_im).astype(bool)
            n_lymphnode = np.count_nonzero(lymphnode_data)
            if n_lymphnode == 0:
                print("Empty groundtruth mask!")
                continue

            for i in range(105):
                ts_foreground_mask = ts_data == i
                all_results[i].append(
                    float(
                        np.sum(
                            np.logical_and(ts_foreground_mask, lymphnode_data),
                            dtype=float,
                        )
                        / float(n_lymphnode)
                    )
                )

        mean_res = {}
        for k, v in all_results.items():
            mean_res[k] = float(np.mean(v))

        non_zero_mean = {}
        for k, v in all_results.items():
            non_zero_mean[k] = float(
                np.mean(
                    np.mean([x for x in v if x != 0])
                    if len([x for x in v if x != 0]) != 0
                    else 0
                )
            )

        out_file = {
            "mean_res": mean_res,
            "non_zero_mean": non_zero_mean,
            "all_results": all_results,
        }
        save_json(out_file, str(out_path))

    return mean_res, non_zero_mean, all_results


def create_original_lnq_dataset(
    train_image_label_paths: list[Path],
    groundtruth_image_paths: list[Path],
    output_path: Path,
    dataset_name: str,
    dataset_json: dict,
):
    """Does not convert or anythin. Just moves files to the correct folder."""
    train_path = output_path / dataset_name / "imagesTr"
    label_path = output_path / dataset_name / "labelsTr"

    train_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    for train_im, train_label in zip(train_image_label_paths, groundtruth_image_paths):
        case_id = int(train_im.name.split("-")[-2])

        shutil.copy(train_im, train_path / (f"{case_id:04}_0000.nrrd"))
        shutil.copy(train_label, label_path / (f"{case_id:04}.nrrd"))

    save_json(dataset_json, output_path / dataset_name / "dataset.json")
    return


def convert_val_samples(val_dir: Path, val_out_dir: Path):
    all_files = [v for v in val_dir.iterdir() if v.name.endswith(".nrrd")]
    convert(all_files, val_out_dir, True)
    return


def create_nnunet_dataset(
    train_image_path: Path,
    groundtruth_image_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_json,
):
    """Moves the data from the chosen paths"""
    train_ids: dict[str, Path] = get_train_ids_and_im_path_from_dir(train_image_path)
    groundtruth_ids: dict[str, Path] = get_ids_and_path_from_dir(groundtruth_image_path)

    train_path = output_path / dataset_name / "imagesTr"
    label_path = output_path / dataset_name / "labelsTr"

    train_path.mkdir(parents=True, exist_ok=True)
    label_path.mkdir(parents=True, exist_ok=True)

    for ids in train_ids.keys():
        train_im = train_ids[ids]
        label_im = groundtruth_ids[ids]

        shutil.copy(train_im, train_path / f"{ids:04}_0000.nii.gz")
        shutil.copy(label_im, label_path / f"{ids:04}.nii.gz")
    save_json(dataset_json, output_path / dataset_name / "dataset.json")
    return


def simple_multidim_isin(arr1: np.ndarray, values: Sequence[int]):
    """Checks if all elements of arr1 are in arr2. arr1 can have more dimensions than arr2."""
    all_masks = []
    for val in values:
        all_masks.append(arr1 == val)
    mask = np.sum(np.stack(all_masks), axis=0) != 0
    return mask


def create_convex_hull_lung_mask(total_segmentator_groundtruth: np.ndarray):
    """Creates the convex hull of the lung, setting everthing inside to 1."""

    left_lung = [13, 14]
    right_lung = [15, 16, 17]

    left_lung_mask = simple_multidim_isin(total_segmentator_groundtruth, left_lung)
    right_lung_mask = simple_multidim_isin(total_segmentator_groundtruth, right_lung)

    joint_lung_mask = np.logical_or(left_lung_mask, right_lung_mask)
    convex_lung_mask, _ = flood_fill_hull(joint_lung_mask)
    # We want the minimum and the maximum along each dimension.
    # Assume that z axis is 0
    """
    left_non_zero_z_ids = np.argwhere(np.sum(left_lung_mask, axis=[1,2]) != 0)  # Only leaves z axis
    right_non_zero_z_ids = np.argwhere(np.sum(right_lung_mask, axis=[1,2]) != 0)  # Only leaves z axis

    minimum_joint_z_score = max(min(left_non_zero_z_ids), min(right_non_zero_z_ids))
    maximum_joint_z_score = min(max(left_non_zero_z_ids), max(right_non_zero_z_ids))

    non_convex_lung_mask = np.zeros_like(total_segmentator_groundtruth)    
    for z in range(minimum_joint_z_score, maximum_joint_z_score):
        slice = joint_lung_mask[z]
        convex_hull = sk_morphology.convex_hull_image(slice)
        non_convex_lung_mask[z] = np.logical_not(convex_hull)  # 1 Where Convex hull is not, 0 where it is. (0 is ignore label later)
    """
    return convex_lung_mask


def flood_fill_hull(image):
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices])
    idx = np.stack(np.indices(image.shape), axis=-1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img, hull


def create_ribcage_convex_hull(total_segmentator_groundtruth: np.ndarray):
    left_ribs = [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
    right_ribs = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81]

    left_ribs_mask = simple_multidim_isin(total_segmentator_groundtruth, left_ribs)
    right_ribs_mask = simple_multidim_isin(total_segmentator_groundtruth, right_ribs)

    joint_rib_mask = np.logical_or(left_ribs_mask, right_ribs_mask)

    # We want the minimum and the maximum along each dimension.
    # Assume that z axis is 0

    convex_ribcage = flood_fill_hull(joint_rib_mask)

    return convex_ribcage


def create_groundtruth_given_totalsegmentator(
    total_segmentator_dir: Path,
    total_segmentator_background_class_ids: Sequence[int],
    groundtruth_dir: Path,
    output_dir: Path,
    overwrite=False,
    make_outside_boundary_class=False,
) -> dict[str, int]:
    """Creates a groundtruth mask given the total segmentator segmentation and the ids of the background classes.
    Returns the dataset.json file."""
    all_total_segmentator_files = list(sorted(os.listdir(total_segmentator_dir)))
    all_groundtruth_files = list(sorted(os.listdir(groundtruth_dir)))
    output_dir.mkdir(exist_ok=True, parents=True)

    if make_outside_boundary_class:
        labels = {
            "background": 0,
            "lymphnode": 1,
            "lymphnode_outside_boundary": 2,
            "ignore": 3,
        }
    else:
        labels = {
            "background": 0,
            "lymphnode": 1,
            "ignore": 2,
        }

    for ts, gt in tqdm(zip(all_total_segmentator_files, all_groundtruth_files)):
        # Read total Segmentator segmentations
        output_path = output_dir / gt

        if output_path.exists():
            if not overwrite:
                continue

        ts_im = sitk.ReadImage(str(total_segmentator_dir / ts))
        ts_data = sitk.GetArrayFromImage(ts_im)
        ts_data = ts_data.astype(int)

        # Read groundtruth
        lnq_im = sitk.ReadImage(str(groundtruth_dir / gt))
        lnq_data = sitk.GetArrayFromImage(lnq_im)
        lnq_data = lnq_data.astype(int)

        # Create final groundtruth (all total_segmentator_background_class_ids + 1 voxel from boundary are background, all lymph nodes are foreground, rest is ignore)
        final_groundtruth = np.full_like(
            lnq_data, fill_value=labels["ignore"]
        )  # 0 will be background, 1 is ignore label and 2 is foreground (lymph node)

        total_segmentator_mask = simple_multidim_isin(
            ts_data, total_segmentator_background_class_ids
        )  # This is slow but running it once is enough, so who cares

        total_segmentator_lung_region = create_outside_lung_axial_mask()

        # Set all total segmentator predicted classes (that we want to set to background) to background
        final_groundtruth = np.where(
            total_segmentator_mask, labels["background"], final_groundtruth
        )
        # To be safe of overlaps we set the lymph node (foreground) after to foreground
        final_groundtruth = np.where(
            lnq_data != 0, labels["lymphnode"], final_groundtruth
        )

        final_groundtruth = final_groundtruth.reshape(lnq_data.shape)

        lnq_binary = np.where(lnq_data != 0, 1, 0)  # Make binary
        dilated_lnq_binary = morphology.binary_dilation(
            lnq_binary, iterations=2
        )  # Dilate with Square connectivity equal to one
        lnq_boundary = (
            dilated_lnq_binary - lnq_binary
        )  # This is the boundary of the object

        if make_outside_boundary_class:
            final_groundtruth = np.where(
                lnq_boundary != 0,
                labels["lymphnode_outside_boundary"],
                final_groundtruth,
            )
        else:
            final_groundtruth = np.where(
                lnq_boundary != 0, labels["background"], final_groundtruth
            )

        output_im = sitk.GetImageFromArray(final_groundtruth.astype(np.uint32))
        output_im.CopyInformation(lnq_im)
        sitk.WriteImage(output_im, str(output_path))

    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": len(all_total_segmentator_files),
        "file_ending": ".nii.gz",
    }

    return dataset_json


def create_convex_hulls(filepath: Path, ribcage_out: Path, lung_out: Path):
    filename = filepath.name
    if not (ribcage_out / filename).exists():
        im = sitk.ReadImage(filepath)
        data = sitk.GetArrayFromImage(im)
        rib_convex_hull, _ = create_ribcage_convex_hull(data)
        rib_convex_im = sitk.GetImageFromArray(rib_convex_hull.astype(np.uint32))
        rib_convex_im.CopyInformation(im)
        sitk.WriteImage(rib_convex_im, str(ribcage_out / filename))
    if not (lung_out / filename).exists():
        im = sitk.ReadImage(filepath)
        data = sitk.GetArrayFromImage(im)
        lung_convex_hull = create_convex_hull_lung_mask(data)
        lung_convex_im = sitk.GetImageFromArray(lung_convex_hull.astype(np.uint32))
        lung_convex_im.CopyInformation(im)
        sitk.WriteImage(lung_convex_im, str(lung_out / filename))
    return

def mp_create_convex_hulls_given_totalsegmentator(
    totalseg_dir: Path, ribcage_out: Path, lung_out: Path
):
    """
    Create the convex hulls
    """
    all_content = [f for f in totalseg_dir.iterdir()]
    ribcage_out.mkdir(exist_ok=True, parents=True)
    lung_out.mkdir(exist_ok=True, parents=True)

    from multiprocessing import Pool
    from functools import partial

    partial_create_convex_hull = partial(create_convex_hulls, ribcage_out=ribcage_out, lung_out=lung_out)

    with Pool(32) as p:
        p.map(partial_create_convex_hull, all_content)
    return

def create_convex_hulls_given_totalsegmentator(
    totalseg_dir: Path, ribcage_out: Path, lung_out: Path
):
    """
    Create the convex hulls
    """
    all_content = [f for f in totalseg_dir.iterdir()]
    ribcage_out.mkdir(exist_ok=True, parents=True)
    lung_out.mkdir(exist_ok=True, parents=True)
    for c in tqdm(all_content):
        filename = c.name
        if not (ribcage_out / filename).exists():
            im = sitk.ReadImage(c)
            data = sitk.GetArrayFromImage(im)
            rib_convex_hull, _ = create_ribcage_convex_hull(data)
            rib_convex_im = sitk.GetImageFromArray(rib_convex_hull.astype(np.uint32))
            rib_convex_im.CopyInformation(im)
            sitk.WriteImage(rib_convex_im, str(ribcage_out / filename))
        if not (lung_out / filename).exists():
            im = sitk.ReadImage(c)
            data = sitk.GetArrayFromImage(im)
            lung_convex_hull = create_convex_hull_lung_mask(data)
            lung_convex_im = sitk.GetImageFromArray(lung_convex_hull.astype(np.uint32))
            lung_convex_im.CopyInformation(im)
            sitk.WriteImage(lung_convex_im, str(lung_out / filename))
    return


def measure_volume_contained_in_convex_hull(
    groundtruth_dir: Path, convex_hull_dir: Path
):
    """Loads same images, and measure how much of label 1 (foreground) of groundtruth is contained in lung/ribcage convex hull (label 1) ."""
    groundtruth_dict = {k.name.split("-")[-2]: k for k in groundtruth_dir.iterdir()}
    convex_hull_dict = {k.name.split("-")[-2]: k for k in convex_hull_dir.iterdir()}

    n_cases = 0
    all_foreground_ratios = []
    non_zero_pct = []
    for k in groundtruth_dict:
        gt_im = sitk.GetArrayFromImage(sitk.ReadImage(str(groundtruth_dict[k])))
        convex_im = sitk.GetArrayFromImage(sitk.ReadImage(str(convex_hull_dict[k])))

        foreground_ratio_in_case = np.sum(
            np.logical_and(gt_im == 1, convex_im == 1)
        ) / np.sum(gt_im == 1)
        if foreground_ratio_in_case != 1:
            n_cases += 1
            non_zero_pct.append(foreground_ratio_in_case)
        all_foreground_ratios.append(foreground_ratio_in_case)
    return n_cases, float(np.mean(all_foreground_ratios)), float(np.mean(non_zero_pct))


def main():
    """
    DISCLAIMER:

    This is a necessary preprocessing step as the annotation scheme of the LNQ challenge is so weird.
    So in order to get some (guaranteed) background class voxels inferences is done of total segmentator.
    Predicted classes that should not overlap with the lymph nodes are used to create actual background classes.

    Additionally to that the Lymphnodes are region-grown (by a little bit) and that area is set as negative clas as well.

    Everything else will be set to ignore label, as it might as well be a lymph node, but we do not know.

    Finally postprocessing will have to deal with the removal of too small lymph nodes as obviously the challenge organizers use the weird 2D radiomics garbage to determine the biggest diameters...

    !!!!!!!!!!
    WARNING: This has to be run in a separate environment to the normal nnUNet prepocessing as TotalSegmentator is incompatible (especially to nnUNet V2!)
    !!!!!!!!!!

    So run this in a totalsegmentator env first, then run the Dataset911_LNQ.py conversion which will need the temporary files created here.
    """

    path_to_data = Path("/dkfz/cluster/gpu/data/OE0441/t006d/LNQ/")
    meta_info_path = path_to_data / "meta_info.json"
    convex_hull_info = path_to_data / "convex_hull_info.json"
    temp_in_path = path_to_data / "total_segmentator_LNQ" / "in"
    temp_lbl_path = path_to_data / "total_segmentator_LNQ" / "lbl"
    temp_out_path = path_to_data / "total_segmentator_LNQ" / "seg"
    ribcage_out_path = path_to_data / "total_segmentator_LNQ" / "ribcage_convex_hull"
    lung_out_path = path_to_data / "total_segmentator_LNQ" / "lung_convex_hull"
    val_path = path_to_data / "val"
    val_nifti_path = path_to_data / "val_nifti"

    convert_val_samples(val_path, val_nifti_path)

    out_dir_default = path_to_data / "background_default"
    out_dir_aorta = path_to_data / "background_no_aorta"
    out_dir_aorta_boundary_class = path_to_data / "background_no_aorta_boundary_class"
    # out_dir_non_zero = path_to_data / "background_non_zero_overlap"
    # out_dir_low_overlap = path_to_data / "background_low_overlap"
    out_dir_medium_overlap = path_to_data / "background_medium_overlap"

    nnunet_raw_data_path = Path(os.environ["nnUNet_raw"])

    train_dir = (
        path_to_data / "patched_train"
    )  # Contains the new segmentation labels that are (hopefully) the same shape as original labels.

    train_ids = get_ids_from_dir(train_dir)

    train_im_labels = [
        get_im_and_label_from_id(id, train_dir, False) for id in train_ids
    ]

    # Assure the train cases and labels have same shape.
    only_train_images = [ti[0] for ti in train_im_labels]
    only_train_labels = [tl[1] for tl in train_im_labels]
    remaining_cases, remaining_labels = filter_non_same_size_cases(
        only_train_images, only_train_labels
    )

    convert(remaining_cases, temp_in_path)
    convert(remaining_labels, temp_lbl_path)
    total_segmentator_predict_dir(temp_in_path, temp_out_path)
    print("Creating convex hulls given Totalsegmentator")
    mp_create_convex_hulls_given_totalsegmentator(
        temp_out_path, ribcage_out_path, lung_out_path
    )

    if not convex_hull_info.exists():
        (
            ribcage_n_cases,
            ribcage_avg_casewise_pct,
            ribcage_non_zero_pct,
        ) = measure_volume_contained_in_convex_hull(temp_lbl_path, ribcage_out_path)
        (
            lung_n_cases,
            lung_avg_casewise_pct,
            lung_non_zero_pct,
        ) = measure_volume_contained_in_convex_hull(temp_lbl_path, lung_out_path)
        save_json(
            {
                "ribcage": {
                    "n_cases": ribcage_n_cases,
                    "avg_casewise_pct": ribcage_avg_casewise_pct,
                    "non_zero_pct": ribcage_non_zero_pct,
                },
                "lung": {
                    "n_cases": lung_n_cases,
                    "avg_casewise_pct": lung_avg_casewise_pct,
                    "non_zero_pct": lung_non_zero_pct,
                },
            },
            str(convex_hull_info),
        )
        print("Finished measuring convex hulls")

    (
        mean_res,
        non_zero_mean,
        all_results,
    ) = find_mutually_exclusive_classes_with_total_segmentator(
        temp_out_path, temp_lbl_path, meta_info_path
    )

    background_classes_no_aorta = {
        int(k) for k, v in non_zero_mean.items() if not (int(k) in [0, 7])
    }  # 0 is background, so we do not want that and do not want classes that overlap with lymphnodes.
    background_classes_medium_overlap = {
        int(k) for k, v in non_zero_mean.items() if (v < 0.1 and k != 0)
    }  # 0 is background, so we do not want that and do not want classes that overlap with lymphnodes.

    # background_classes_non_zero = {
    #     int(k) for k, v in mean_res.items() if (v == 0 and k != 0)
    # }  # 0 is background, so we do not want that and do not want classes that overlap with lymphnodes.
    # background_classes_low_overlap = {
    #     int(k) for k, v in non_zero_mean.items() if (v < 0.01 and k != 0)
    # }  # 0 is background, so we do not want that and do not want classes that overlap with lymphnodes.

    print("No Aorta: Create groundtruth given total segmentator")
    no_aorta_dataset_json = create_groundtruth_given_totalsegmentator(
        temp_out_path, background_classes_no_aorta, temp_lbl_path, out_dir_aorta
    )

    print(
        "No Aorta with boundary class dataset: Create groundtruth given total segmentator"
    )
    no_aorta_with_boundary_class_dataset_json = (
        create_groundtruth_given_totalsegmentator(
            temp_out_path,
            background_classes_no_aorta,
            temp_lbl_path,
            out_dir_aorta_boundary_class,
            make_outside_boundary_class=True,
        )
    )

    print("Medium overlap classes: Create groundtruth given total segmentator")
    medium_overlap_dataset_json = create_groundtruth_given_totalsegmentator(
        temp_out_path,
        background_classes_medium_overlap,
        temp_lbl_path,
        out_dir_medium_overlap,
    )
    # create_groundtruth_given_totalsegmentator(
    #     temp_out_path, background_classes_non_zero, temp_lbl_path, out_dir_non_zero
    # )
    # create_groundtruth_given_totalsegmentator(
    #     temp_out_path,
    #     background_classes_low_overlap,
    #     temp_lbl_path,
    #     out_dir_low_overlap,
    # )
    # Now we create the nnUNet compatible datasets

    print("Create Dataset911_LNQ_original")
    create_original_lnq_dataset(
        only_train_images,
        only_train_labels,
        nnunet_raw_data_path,
        "Dataset911_LNQ_original",
        dataset_json={
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "lymphnode": 1},
            "numTraining": len(only_train_images),
            "file_ending": ".nrrd",
        },
    )

    print("Create Dataset912_aorta_not_background")
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_aorta,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset912_aorta_not_background",
        dataset_json=no_aorta_dataset_json,
    )

    print("Create Dataset913_aorta_not_background_with_boundary_class.")
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_aorta_boundary_class,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset913_aorta_not_background_with_boundary_class",
        dataset_json=no_aorta_with_boundary_class_dataset_json,
    )

    print("Create Dataset914_medium_overlap_not_background.")
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_medium_overlap,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset914_medium_overlap_not_background",
        dataset_json=medium_overlap_dataset_json,
    )

    # create_nnunet_dataset(
    #     train_image_path=temp_in_path,
    #     groundtruth_image_path=out_dir_low_overlap,
    #     output_path=path_to_data / "nnunet",
    #     dataset_name="low-overlap-not-background",
    # )

    # create_nnunet_dataset(
    #     train_image_path=temp_in_path,
    #     groundtruth_image_path=out_dir_non_zero,
    #     output_path=path_to_data / "nnunet",
    #     dataset_name="non-zero-overlap-not-background",
    # )

    sys.exit(0)


if __name__ == "__main__":
    main()
