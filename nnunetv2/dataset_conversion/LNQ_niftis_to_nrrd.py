import SimpleITK as sitk
import os
from pathlib import Path

def sitk_convert(in_file: str, out_file: str):
    """Convert nrrd file to nifti file."""
    im = sitk.ReadImage(in_file)
    sitk.WriteImage(im, out_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    args = parser.parse_args()

    input_path = Path(args.i)
    output_path = Path(args.o)
    
    print(f"Checking input path: {input_path}")
    niftis_of_interest = [i for i in input_path.iterdir() if i.name.endswith(".nii.gz")]
    print(f"Found {len(niftis_of_interest)} niftis of interest.")
    for input_file in niftis_of_interest:
        output_name = input_file.name[:-9] + "seg.nrrd"  # Removes the 'ct.nii.gz'
        output_file = output_path / output_name
        sitk_convert(input_file, output_file)

if __name__ == "__main__":
    main()


    