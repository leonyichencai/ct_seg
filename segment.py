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

import subprocess
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
    weights_file = '/Users/snaplab/Projects/ct_seg/weights.hdf5'
    tmp_file = out_file.replace('.nii', '__tmp.nii')

    # Load images

    print('segment.py: Loading inputs...')
    print('segment.py: - Input:       {}'.format(in_file))
    print('segment.py: - Output/temp: {}'.format(out_file))

    nii = nib.load(in_file)
    img = nii.get_fdata()
    aff = nii.affine
    hdr = nii.header
    
    ivox = img.shape[0]
    jvox = img.shape[1]
    kslices = img.shape[2]

    regrid = not ivox == 512 and not jvox == 512

    # Regrid input as indicated

    if regrid:
        print('segment.py: Regridding inputs from {}x{}x{} to 512x512x{}...'.format(ivox, jvox, kslices, kslices))
        regrid_cmd = 'mrgrid {} regrid {} -size 512,512,{} -interp linear -force'.format(in_file, tmp_file, kslices)
        subprocess.check_call(regrid_cmd, shell=True)
        nii = nib.load(tmp_file)
        img = nii.get_fdata()

    # Format for model

    print('segment.py: Reformatting for model...')
    img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, -1).astype(np.float32)

    # Prepare model

    print('segment.py: Loading weights from {}...'.format(weights_file))
    nb_classes = 17 # number of classes (+1 for background)
    model = unet(nb_classes, None, True)
    model.load_weights(weights_file)

    # Run model

    print('segment.py: Segmenting input...')
    seg = model.predict(img, batch_size=4, verbose=1)

    # Save output

    print('segment.py: Saving output...')
    seg = seg.reshape((kslices, 512, 512, nb_classes))
    seg = np.argmax(seg, axis=3)
    seg = np.moveaxis(seg, 0, -1).astype('uint16')
    out = nib.Nifti1Image(seg, nii.affine)
    nib.save(out, tmp_file)

    # Regrid back if needed

    if regrid:
        print('segment.py: Regridding output from 512x512x{} to {}x{}x{}...'.format(kslices, ivox, jvox, kslices))
        regrid_cmd = 'mrgrid {} regrid {} -size {},{},{} -interp nearest -force'.format(tmp_file, out_file, ivox, jvox, kslices)
        subprocess.check_call(regrid_cmd, shell=True)
    else:
        print('segment.py: Renaming temp to output...')
        cp_cmd = 'cp {} {}'.format(tmp_file, out_file)
        subprocess.check_call(cp_cmd, shell=True)

    # Clean up

    print('segment.py: Copying header from input to output...')
    cp_cmd = 'fslcpgeom {} {}'.format(in_file, out_file)
    subprocess.check_call(cp_cmd, shell=True)

    print('segment.py: Removing temp files...')
    rm_cmd = 'rm {}'.format(tmp_file)
    subprocess.check_call(rm_cmd, shell=True)

    print('segment.py: Done!')