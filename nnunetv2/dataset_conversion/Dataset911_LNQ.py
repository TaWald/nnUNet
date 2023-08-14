from pathlib import Path
import os
import shutil
import sys
import time
from typing import Sequence
import nrrd
import numpy as np
from tqdm import tqdm
import vtk
import SimpleITK as sitk
from scipy.ndimage import morphology
from batchgenerators.utilities.file_and_folder_operations import save_json, load_json
import torch
from torch.cuda import OutOfMemoryError


from totalsegmentator.python_api import totalsegmentator


def readnrrd(filename):
    """Read image in nrrd format."""
    reader = vtk.vtkNrrdReader()
    reader.SetFileName(filename)
    reader.Update()
    info = reader.GetInformation()
    return reader.GetOutput(), info


def writenifti(image, filename, info):
    """Write nifti file."""
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetInputData(image)
    writer.SetFileName(filename)
    writer.SetInformation(info)
    writer.Write()


def convert_nrrd_to_nifti(nrrd_file: str, nifti_file: str):
    """Convert nrrd file to nifti file."""
    image, info = readnrrd(nrrd_file)
    writenifti(image, nifti_file, info)


def convert(input_image: list[Path], output_path: Path):
    """Convert all nrrd files in input_images to nifti files in output_path."""
    output_path.mkdir(exist_ok=True, parents=True)
    for im in tqdm(input_image):
        im_name = im.name
        nifti_name = im_name[:-5] + ".nii.gz"
        if (output_path / nifti_name).exists():
            continue
        else:
            convert_nrrd_to_nifti(str(im), str(output_path / nifti_name))


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
            totalsegmentator(str(im), str(output_dir / out_name), nr_thr_saving=1, ml=True, fast=False, force_split=False)


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
            assert (
                ts.split("-")[2] == gt.split("-")[2]
            ), "Total segmentator and groundtruth files do not match!"

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


def create_nnunet_dataset(
    train_image_path: Path,
    groundtruth_image_path: Path,
    output_path: Path,
    dataset_name: str,
    dataset_json
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
    save_json(dataset_json, output_path / "dataset.json")
    return


def multidim_isin(arr1: np.ndarray, values: Sequence[int]):
    """Checks if all elements of arr1 are in arr2. arr1 can have more dimensions than arr2."""
    org_shape = arr1.shape
    flat_arr1 = arr1.ravel()
    flat_bin_mask = np.isin(flat_arr1, values)
    return flat_bin_mask.reshape(org_shape)


def test_multidim_isin(arr1: np.ndarray, values: Sequence[int]):
    """Checks if all elements of arr1 are in arr2. arr1 can have more dimensions than arr2."""
    all_masks = []
    for val in values:
        all_masks.append(arr1 == val)
    mask = np.sum(np.stack(all_masks), axis=0) != 0
    return mask


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

        # Flatten ts_data and final_groundtruth then check if the values of ts_data are in total_segmentator_background_class_ids.
        #  If so, set the value to 0 there and else to the value in the flat final_groundtruth array.
        #  Then reshape the final_groundtruth array to the shape of the original lnq_data array.
        # mask = multidim_isin(ts_data, total_segmentator_background_class_ids)
        total_segmentator_mask = test_multidim_isin(
            ts_data, total_segmentator_background_class_ids
        )  # This is slow but running it once is enough, so who cares

        # assert np.all(mask == mask_1), "Masks are not equal!"

        # Set all total segmentator predicted classes (that we want to set to background) to background
        final_groundtruth = np.where(total_segmentator_mask, labels["background"], final_groundtruth)  
        # To be safe of overlaps we set the lymph node (foreground) after to foreground
        final_groundtruth = np.where(lnq_data != 0, labels["lymphnode"], final_groundtruth)

        final_groundtruth = final_groundtruth.reshape(lnq_data.shape)

        lnq_binary = np.where(lnq_data != 0, 1, 0)  # Make binary
        dilated_lnq_binary = morphology.binary_dilation(
            lnq_binary, iterations=2
        )  # Dilate with Square connectivity equal to one
        lnq_boundary = dilated_lnq_binary - lnq_binary  # This is the boundary of the object

        if make_outside_boundary_class:
            final_groundtruth = np.where(lnq_boundary != 0, labels["lymphnode_outside_boundary"], final_groundtruth)
        else:
            final_groundtruth = np.where(lnq_boundary != 0, labels["background"], final_groundtruth)

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
    temp_in_path = path_to_data / "total_segmentator_LNQ" / "in"
    temp_lbl_path = path_to_data / "total_segmentator_LNQ" / "lbl"
    temp_out_path = path_to_data / "total_segmentator_LNQ" / "seg"

    out_dir_aorta = path_to_data / "background_no_aorta"
    out_dir_aorta_boundary_class = path_to_data / "background_no_aorta_boundary_class"
    # out_dir_non_zero = path_to_data / "background_non_zero_overlap"
    # out_dir_low_overlap = path_to_data / "background_low_overlap"
    out_dir_medium_overlap = path_to_data / "background_medium_overlap"

    nnunet_raw_data_path = os.environ['raw_data']

    train_dir = path_to_data / "patched_train"  # Contains the new segmentation labels that are (hopefully) the same shape as original labels.

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

    # Create different groundtruths
    no_aorta_dataset_json = create_groundtruth_given_totalsegmentator(
        temp_out_path, background_classes_no_aorta, temp_lbl_path, out_dir_aorta
    )
    no_aorta_with_boundary_class_dataset_json = create_groundtruth_given_totalsegmentator(
        temp_out_path, background_classes_no_aorta, temp_lbl_path, out_dir_aorta_boundary_class, make_outside_boundary_class=True
    )
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
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_aorta,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset912_aorta-not-background",
        dataset_json=no_aorta_dataset_json,
    )
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_aorta_boundary_class,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset913_aorta-not-background-with-boundary-class",
        dataset_json=no_aorta_with_boundary_class_dataset_json,
    )
    create_nnunet_dataset(
        train_image_path=temp_in_path,
        groundtruth_image_path=out_dir_medium_overlap,
        output_path=nnunet_raw_data_path,
        dataset_name="Dataset914_medium-overlap-not-background",
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
