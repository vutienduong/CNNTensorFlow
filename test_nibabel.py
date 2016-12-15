import numpy as np
np.set_printoptions(precision=2, suppress=True)
import os
import nibabel as nib
import Tkinter
import matplotlib.pyplot as plt

from nibabel.testing import data_path

# Show slices function
def show_slices(slices):
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		print i
		axes[i].imshow(slice.T, cmap="gray", origin="lower")


path = "/media/sf_LEARN/4th SEMESTER/ADNI199/ADNI/002_S_0559/MPR____N3__Scaled/2006-06-27_18_28_33.0/S15922";
file_name = "ADNI_002_S_0559_MR_MPR____N3__Scaled_Br_20070319121214158_S15922_I45126.nii";
# example_ni1 = os.path.join(data_path, 'example4d.nii.gz')
example_ni1 = os.path.join(path, file_name)
n1_img = nib.load(example_ni1)
n1_img
n1_header = n1_img.header
# print n1_header
# print "wo"
# print n1_header['cal_max']
# print n1_img.affine
# print(n1_header['srow_x'])
# print(n1_header.get_sform())
# print(n1_header['sform_code'])
# print (n1_header.get_data_shape())
# print (n1_header.get_data_dtype())
# print (n1_header.get_zooms())

## PROXY
# print (n1_img.dataobj)
# print nib.is_proxy(n1_img.dataobj)

## ARRAY IMAGES
# array_data = np.arange(24, dtype=np.int16).reshape((2, 3, 4))
# affine = np.diag([1, 2, 3, 1])
# array_img = nib.Nifti1Image(array_data, affine)
# print array_img.dataobj
# print nib.is_proxy(array_img.dataobj)

## EASY WAY
image_data = n1_img.get_data()
print image_data.shape
# print image_data
slice_0 = image_data[125, :, :]
slice_1 = image_data[:, 125, :]
slice_2 = image_data[:, :, 80]
show_slices([slice_0, slice_1, slice_2])
plt.show()
plt.suptitle("Center slices for EPI image")  