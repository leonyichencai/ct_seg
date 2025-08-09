# Deep learning-based 17-class CT head segmentation
# Leon Cai
# August 8, 2025

# Modified from:

##################################################################################
##  (c) 2020                                                                    ##
##  Radiology Informatics Lab                                                   ##
##  Department of Radiology                                                     ##
##  Mayo Clinic Rochester                                                       ##
## ---------------------------------------------------------------------------- ##
##  Source code for the manuscript entitled:                                    ##
##  "Fully automated segmentation of head CT neuroanatomy using deep learning"  ##
##  https://doi.org/10.1148/ryai.2020190183                                     ##
##  Code has been updated to Tensorflow 2                                       ##
##################################################################################

# Set Up

import argparse as ap
import numpy as np
import nibabel as nib
from z_unet import unet

# Go!

if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Inference wrapper for HeadCTSegmentation by Cai et al. (2020).')
    parser.add_argument('in_file', help='Path to the input NIFTI non-contrasted CT head')
    parser.add_argument('out_file', help='Path to save the segmentation (careful, destructive!)')
    args = parser.parse_args()

    # Get inputs

    in_file = args.in_file
    out_file = args.out_file
    weights_file = '/Users/snaplab/Projects/ct_seg/HeadCTSegmentation/weights.hdf5'

    # Prepare model

    nb_classes = 17 # number of classes (+1 for background)
    model = unet(nb_classes, None, True)
    model.load_weights(weights_file)

    # Load images

    nii = nib.load(in_file)
    img = nii.get_fdata()
    nslices = img.shape[2]
    img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, -1).astype(np.float32)

    # Run model

    seg = model.predict(img, batch_size=4, verbose=1)

    # Save output

    seg = seg.reshape((nslices, 512, 512, nb_classes))
    seg = np.argmax(seg, axis=3)
    seg = np.moveaxis(seg, 0, -1).astype('uint16')
    out = nib.Nifti1Image(seg, nii.affine, nii.header)
    nib.save(out, out_file)