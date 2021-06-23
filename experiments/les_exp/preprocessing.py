'''
PREPROCESS
This script preprocess our input data. The main action is the resample and
center-cropping of the 3d-images. The output is a pickle file with all the 
meta-data of the scans and lesions, and a npy file which contains the preprocessed 
scans and masks with the shape:
    [Sc, W, H, Sl] : Sl - slice
                     Sc - scans (T2W, ADC, b400, b50, b800, Ktrans, pros, pz, tz, les)

Ben Arnon, 2021
'''


# imports
import os
import sys
import pathlib
import numpy as np
import glob
import pickle
import SimpleITK as sitk
import pprint
pp = pprint.PrettyPrinter(indent=0)

# settings
script_path = str(pathlib.Path(__file__).parent.absolute())
PATH = '/data/bigdata/fuse/data'
OUT_PATH = script_path + '/preprocessing_out'

if not os.path.exists(OUT_PATH): os.makedirs(OUT_PATH)
CROP_SIZE_PX = tuple((160, 160, 24))
CROP_SIZE_MM = tuple((80, 80, 72))
CROP_SPACING = tuple(([CROP_SIZE_MM[i] / CROP_SIZE_PX[i] for i in range(len(CROP_SIZE_MM))]))


# functions
def rescale_intensity(image, thres=(1.0, 99.0), method='noclip'):
    '''
        Rescale the image intensity using several possible ways
        
        Parameters
        ----------
        image: array
            Image to rescale
        thresh: list of two floats between 0. and 1., default (1.0, 99.0)
            Percentiles to use for thresholding (depends on the `method`)
        method: str, one of ['clip', 'mean', 'median', 'noclip']
            'clip': clip intensities between the thresh[0]th and the thresh[1]th
            percentiles, and then scale between 0 and 1
            'mean': divide by mean intensity
            'meadin': divide by meadian intensity
            'noclip': Just like 'clip', but wihtout clipping the extremes
            
        Returns
        -------
        image: array
    '''
    eps= 0.000001
    def rescale_single_channel_image(image):
        #Deal with negative values first
        min_value= np.min(image)
        if min_value < 0:
            image-= min_value
        if method == 'clip':
            val_l, val_h = np.percentile(image, thres)
            image2 = image
            image2[image < val_l] = val_l
            image2[image > val_h] = val_h
            image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        elif method == 'mean':
            image2= image / max(np.mean(image),1)
        elif method == 'median':
            image2= image / max(np.median(image),1)
        elif method == 'noclip':
            val_l, val_h = np.percentile(image, thres)
            image2 = image
            image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l + eps)
        else:
            image2= image
        return image2
    
    #Process each channel independently
    if len(image.shape) == 4:
        for i in range(image.shape[-1]):
            image[...,i]= rescale_single_channel_image(image[...,i])
    else:
        image= rescale_single_channel_image(image)
        
    return image

def transform_img(img_sitk, ref_img_sitk):
    # https://simpleitk.org/SimpleITK-Notebooks/01_Image_Basics.html        
    # # img_sitk.GetPixelID() == 8 (sitkFloat32)
    # img_sitk.GetPixelID() == 1 (sitkUInt8)
         
    # check if image is float32
    if img_sitk.GetPixelID() == 8:
        interpolation_method = sitk.sitkBSpline
    elif img_sitk.GetPixelID() in [0,1]:
        interpolation_method = sitk.sitkNearestNeighbor
    else:
        print(img_sitk.GetPixelID())
        raise ValueError('wrong pixel type')
        
    # configure resampling filter
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref_img_sitk)
    resample.SetInterpolator(interpolation_method)
    # apply resampling filter
    img_sitk_new = resample.Execute(img_sitk)
    return img_sitk_new

def get_itk_info(img_sitk, scan_type):
    info = {
        scan_type + '_GetOrigin': img_sitk.GetOrigin(),
        scan_type + '_GetDirection': img_sitk.GetDirection(),
        scan_type + '_GetSpacing': img_sitk.GetSpacing(),
        scan_type + '_GetSize': img_sitk.GetSize(),
        scan_type + '_GetDimension': img_sitk.GetDimension()
    }
    return info

def binary_bbox_3D(img_arr):
    x = np.any(img_arr, axis=(0))
    y = np.any(img_arr, axis=(1))
    z = np.any(img_arr, axis=(2))
    try:
        xmin, xmax = np.where(x)[0][[0, -1]]
        ymin, ymax = np.where(y)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        return [xmin-1, ymin-1, xmax+1, ymax+1, zmin-1, zmax+1]
    except:
        print('ERROR: could not retrive bbox:', p, '\n', les)
        return [0,0,0,0,0,0]

def reset_sitk(img_sitk):
    img_sitk.SetOrigin((0,)*3)
    img_sitk.SetDirection(np.eye(3).flatten())
    return img_sitk

def resampling(img_sitk, transform, scan_type, interpolation=sitk.sitkBSpline):    
    # (sitkBSpline, sitkNearestNeighbor, sitkLabelGaussian, sitkLinear, sitkCosineWindowedSinc)
    resample = sitk.ResampleImageFilter()
    if 'mask' in scan_type:
        print(scan_type)
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(interpolation)
        print(scan_type)
    resample.SetOutputSpacing(CROP_SPACING)
    resample.SetSize(CROP_SIZE_PX)
    resample.SetOutputOrigin(img_sitk.GetOrigin())
    resample.SetOutputDirection(img_sitk.GetDirection())
    resample.SetTransform(transform)
    img_sitk = resample.Execute(img_sitk)
    img_arr = sitk.GetArrayFromImage(img_sitk)
    return np.expand_dims(img_arr, axis=-1)


# list patients and paths
imgs_nrrd = glob.glob(os.path.join(PATH, 'prosx-[0-9][0-9][0-9][0-9]_*_img_prosx_rad.nrrd'))
pat_list = list(set([p.split('/')[-1].split('_')[0] for p in imgs_nrrd]))
pat_list.sort()

# iterate over patients
for p in pat_list:
    print(p)
    # read prostate segmentation masks of patient p
    mask_nrrd = glob.glob(os.path.join(PATH, p + "_t2w_pro_prosx_rad.nrrd"))[0]
    mask_sitk = sitk.ReadImage(mask_nrrd)
    pz_nrrd = glob.glob(os.path.join(PATH, p + "_t2w_zon-pz_prosx_rad.nrrd"))[0]
    pz_sitk = sitk.ReadImage(pz_nrrd)
    tz_nrrd = glob.glob(os.path.join(PATH, p + "_t2w_zon-tz_prosx_rad.nrrd"))[0]
    tz_sitk = sitk.ReadImage(tz_nrrd)

    # list all lesions of a patient p 
    les_nrrd = glob.glob(os.path.join(PATH, p + "_t2w_les-*_prosx_rad*.nrrd"))

    # list all scans of a patient p in order [T2W, ADC, b400, b50, b800, Ktrans]
    b_list = glob.glob(os.path.join(PATH, p + '_b-*.nrrd'))
    scans_list = [i for i in imgs_nrrd+b_list if p in i]
    scans_list.sort()
    scans_list.insert(0, scans_list.pop(-1))

    # create empty npy place holder and info dict to append all scans and mask into
    out_arr = None
    pat_dict = {
        'resampled_GetSpacing': CROP_SPACING,
        'resampled_GetSize': CROP_SIZE_PX,
        'resampled_GetOrigin': ((0,)*3),
        'resampled_GetDirection': tuple((np.eye(3).flatten()))
    }
    
    # save original mask data into the patient's dict
    pat_dict.update(get_itk_info(mask_sitk, 'mask_pro'))
    pat_dict.update(get_itk_info(tz_sitk, 'mask_tz'))
    pat_dict.update(get_itk_info(pz_sitk, 'mask_pz'))

    t2w_ref_sitk = sitk.ReadImage(scans_list[0])
    # mask_sitk = transform_img(mask_sitk, t2w_ref_sitk)
    # tz_sitk = transform_img(tz_sitk, t2w_ref_sitk)
    # pz_sitk = transform_img(pz_sitk, t2w_ref_sitk)

    # iterate every scan; then process and append to the output
    for scan in scans_list:
        name = scan.split('/')[-1].split('_')[1]
        img_sitk = sitk.ReadImage(scan)

        # save original image data into the patient's dict
        pat_dict.update(get_itk_info(img_sitk, name))

        if name != 't2w':
            img_sitk = transform_img(img_sitk, t2w_ref_sitk)

        # rescale intensity (must be converted to numpy first)
        img_backup = sitk.Image(img_sitk)
        img_array = sitk.GetArrayFromImage(img_sitk)
        img_array = rescale_intensity(img_array)
        img_sitk = sitk.GetImageFromArray(img_array)
        img_sitk.CopyInformation(img_backup)

        # reset image properties
        img_sitk = reset_sitk(img_sitk)
        mask_sitk = reset_sitk(mask_sitk)

        # get the cropping translation transform from the mask
        if mask_sitk.GetNumberOfComponentsPerPixel() > 1:
            ma_centroid = sitk.VectorIndexSelectionCast(mask_sitk, 0) > 0.5
        else:
            ma_centroid = mask_sitk > 0.5
        label_analysis_filer = sitk.LabelShapeStatisticsImageFilter()
        label_analysis_filer.Execute(ma_centroid)
        centroid = label_analysis_filer.GetCentroid(1)
        offset_correction = np.array(CROP_SIZE_PX)*np.array(CROP_SPACING)/2
        offset = np.array(centroid)-np.array(offset_correction)
        if name == 't2w':
            pat_dict.update({'offset': offset})
        translation = sitk.TranslationTransform(3, offset)

        # resample, resize, and append the image into the output
        img_arr = resampling(img_sitk, transform=translation, scan_type=name)
        mask_arr = resampling(mask_sitk, transform=translation, scan_type='mask_pro')

        if isinstance(out_arr, np.ndarray):
            out_arr = np.concatenate([out_arr, img_arr], axis=-1)
        else:
            out_arr = img_arr

    # append prostate masks into the output's last channels
    tz_sitk = reset_sitk(tz_sitk)
    pz_sitk = reset_sitk(pz_sitk)
    tz_arr = resampling(tz_sitk, transform=translation, scan_type='mask_tz')
    pz_arr = resampling(pz_sitk, transform=translation, scan_type='mask_pz')
    out_arr = np.concatenate([out_arr, mask_arr, tz_arr, pz_arr], axis=-1)

    # append lesion masks into the output's last channels, add grade and bbox info to dict
    les_arr = np.zeros_like(pz_arr)
    bb_target = []
    class_target = []
    #grade_dict = {'10': 1, '1': 2, '2': 3, '3': 4, '4': 5, '5': 6}
    grade_dict = {'10': 1, '1': 2, '2': 3, '3': 4, '4': 4, '5': 4}
    if len(les_nrrd) > 0:
        les_nrrd.sort()
        for i, les in enumerate(les_nrrd):
            grade = les.split('_')[-3].split('--')[-1].split('-')[-1]
            class_target.append(grade_dict[grade])
            pat_dict.update({'les-'+str(i)+'_grade': grade})
            les_sitk = reset_sitk(sitk.ReadImage(les))
            les_r = resampling(les_sitk, transform=translation, scan_type='mask_les_'+str(i))
            bbox = binary_bbox_3D(les_r)
            pat_dict.update({'les-'+str(i)+'_bbox': bbox})
            bb_target.append(bbox)
            les_r = les_r * (i+1)
            les_arr += les_r
    out_arr = np.concatenate([out_arr, les_arr], axis=-1)
    pat_dict.update({
        'lesions_n': len(les_nrrd),
        'bb_target': np.array(bb_target),
        'class_target': np.array(class_target)
    })
    print('bb_target', np.array(bb_target))
    print('class_target', np.array(class_target))

    # save output array as npy: prosx-<patient_id>_img.npy
    out_arr = np.swapaxes(out_arr, 0, -1)
    np.save(os.path.join(OUT_PATH, p + '_img.npy'), out_arr)
    
    # save patient dictionary as pickle: prosx-<patient_id>_info.pickle
    path_pickle = os.path.join(OUT_PATH, p + '_info.pickle')
    with open(path_pickle, 'wb') as f:
        pickle.dump(pat_dict, f, pickle.HIGHEST_PROTOCOL)

    print()