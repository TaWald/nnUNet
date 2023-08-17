from argparse import ArgumentParser
from pathlib import Path
import SimpleITK as sitk
import shutil

original_data_dir = Path("/dkfz/cluster/gpu/data/OE0441/t006d/nnunetv2/results/lnq_test_data/val_nrrd")


def align_pred_imgs(pred_dir: Path, out_dir: Path):
    out_dir.mkdir(exist_ok=True, parents=True)
    
    for orig_file in original_data_dir.iterdir():
        orig_image_id = orig_file.name.split("_")[0]
        original_im = sitk.ReadImage(str(orig_file))

        potential_nifti_path = (pred_dir / (orig_image_id + '.nii.gz'))
        potential_nrrd_path = (pred_dir / (orig_image_id + '.nrrd'))
        output_path = (out_dir / (orig_image_id[:-2] + 'seg.nrrd'))
        
        if potential_nifti_path.exists():
            pred_img = sitk.ReadImage(str(potential_nifti_path))
            pred_img.CopyInformation(original_im)
        elif potential_nrrd_path.exists():
            pred_img = sitk.ReadImage(str(potential_nrrd_path))
            pred_img.CopyInformation(original_im)
        else:
            raise ValueError(f"Could not find prediction for {orig_image_id}")
        pred_img = sitk.Cast(pred_img, sitk.sitkUInt8)
        sitk.WriteImage(pred_img, str(output_path))


def main():
    parser = ArgumentParser()
    parser.add_argument("-pred", type=Path, required=True)
    parser.add_argument("-out", type=Path, required=True)
    args = parser.parse_args()
    align_pred_imgs(args.pred, args.out)
    # Create an archive of the output folder
    shutil.make_archive(args.out, 'zip', args.out)

if __name__ == "__main__":
    main()