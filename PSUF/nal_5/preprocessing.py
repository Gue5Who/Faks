# %% 1. Nalozimo originalno sliko in pogledamo metapodatke
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt

filepath_CT = "images/0381_1.nii.gz" # pot do CT datoteke

all = nib.load(filepath_CT) # slika in vsi metapodatki
info = all.header # metapodatki se nahajajo v glavi

print(info)

# %% 2. Nalozimo originalno sliko

I_dataobject = all.dataobj # vzamemo le podatkovni objekt

I = np.array(I_dataobject) # pretvorimo v numpy matriko, dimenzij 512x512xNz

plt.figure()
plt.title("Original")
plt.imshow(np.rot90(I[:,:,26], k=3),cmap='gray') # primer vizualizacije 27. rezine (k=3 zaradi rotacija 3x90°)
# %% 3. Sliko normaliziramo

def normalize(image, MIN_BOUND, MAX_BOUND):
	# Normalizacija CT slike znotraj intervala MIN_BOUND, MAX_BOUND.
	image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
	image[image>1] = 1.
	image[image<0] = 0.
	return image

# Uporabimo W = 1500, L = -600 --> https://radiopaedia.org/articles/windowing-ct
nI = normalize(I, -1350, 150) # dobimo normalizirano CT sliko z W = 1500, L = -600

plt.figure()
plt.title("Normalizacija")
plt.imshow(np.rot90(nI[:,:,26], k=3), cmap='gray') # primer vizualizacije 27. rezine segmentacije

# %% 4. Nalozimo masko

import nrrd

filepath_mask = "masks/0381_1.nrrd" # pot do maske

M, _ = nrrd.read(filepath_mask) # preberemo masko in jo shranimo v M

plt.figure()
plt.title("Vizualizacija maske")
plt.imshow(np.rot90(M[:,:,26], k=3), cmap='gray')

# %% 5. Apliciramo masko na normalizirano sliko

S = np.where(M==1, nI, M)

plt.figure()
plt.title("Segmentacija")
plt.imshow(np.rot90(S[:,:,26], k=3),cmap='gray') # primer vizualizacije 27. rezine segmentacije


# %% 6. Popravimo vidno polje
from cv2 import resize

def bound_box(img, slice_idx):
	# Za dano rezino poiscemo najmanjsi pravokotnik, ki obdaja segmentirana pljuca.
	I_slice = img[:,:,slice_idx]
	rows = np.any(I_slice, axis=1)
	cols = np.any(I_slice, axis=0)
	rmin, rmax = np.where(rows)[0][[0, -1]]
	cmin, cmax = np.where(cols)[0][[0, -1]]
	I_slice = I_slice[rmin:rmax, cmin:cmax]
	I_slice = np.transpose(I_slice[:, :, np.newaxis],axes = [2,0,1]).astype('float32')

	return I_slice

bounded_slice = bound_box(S, 26) # npr. da zelimo obrezati 27. rezino S-ja

plt.figure()
plt.title("Popravljeno vidno polje")
plt.imshow(np.rot90(bounded_slice[0,:,:], k=3), cmap='gray')
