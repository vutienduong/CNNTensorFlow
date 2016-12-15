import nibabel as nib


# epi_img = nib.load('someones_epi.nii.gz')
epi_img = nib.load('/media/sf_LEARN/4th SEMESTER/TensorFlow/JH41/resizeMRI/AN_YEONG_SUN_resize30x30x20.nii')
epi_img_data = epi_img.get_data()
print epi_img_data.shape

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

import matplotlib.rcsetup as rcsetup

print(rcsetup.all_backends)

print matplotlib.matplotlib_fname()
print matplotlib.get_backend()

def show_slices(slices):
	fig, axes = plt.subplots(1, len(slices))
	for i, slice in enumerate(slices):
		axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[15, :, :]
slice_1 = epi_img_data[:, 15, :]
slice_2 = epi_img_data[:, :, 10]

show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  

# img=mpimg.imread('games.png')
# imgplot = plt.imshow(img)

# plt.plot(range(20),range(20))
plt.show()