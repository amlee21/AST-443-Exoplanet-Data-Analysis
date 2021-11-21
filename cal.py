# A. Lee 2021-10-30 

# 11/8/21 edit -- using normalized, deblended, minimum distance'ed, ref stars to calculate weighted means of ref. stars
 
# Beginning steps of calibration for science target data (4.5) 

import numpy as np 
from astropy.io import fits 
import matplotlib.pyplot as plt 

# From the manual
# ===============

# Weighted mean: 
# mu_i^{ref} = \sum_j (f_j^_{ref})/(sigma_j^{ref})^2 

# uncertainty on weighted mean: 
# sigma_i^{ref} = \sum_j 1/(sigma_j^{ref})^2 

# reference stars data 
data_list = np.loadtxt('./analysis_files/deblended_norm_ref.txt', dtype='str')

# exptime for each image (acts as our DATE-OBS)
exptime = np.loadtxt('./analysis_files/exptime.dat', dtype='str')
# target star data 
target = np.loadtxt('./analysis_files/deblended_target_star.dat')

print(data_list)

# empty arrays to store numerator and denominator of mu_i^{ref}
num = np.array([])
denom = np.array([]) 

# loop thr. every ref. star file and calculate the weighted mean and uncertainties using every ref. star
for i in range(len(data_list)):
    data = np.loadtxt('./analysis_files/{}'.format((data_list[i]))) # access data 
    flux = data[:,2] 
    flux_avg = np.average(flux)
    print(flux_avg)
    flux_uncert = data[:,3]
    print(np.average(flux_uncert))
    #  for (4.5)
    num = np.append(num, flux/(flux_uncert**2))
    denom = np.append(denom, 1/(flux_uncert**2))

    
# Reshape to be shape (7,319) so that we can sum over them in each column (i.e. over the same image
# in each reference star) to get the weighted mean; each row is a ref. star., each col. is an image 

# 11/8/21 --- edit now do (8,319) instead since we no longer need to throw out any of the ref. stars it seems
 
#print(num_sum, np.shape(num_sum))
#print(denom_sum, np.shape(denom_sum))

num_reshape = np.reshape(num, (8,319))
denom_reshape = np.reshape(denom, (8,319))

num_sum = np.sum(num_reshape, axis=0)
denom_sum = np.sum(denom_reshape, axis=0)

print(np.shape(num_sum), np.shape(denom_sum))
print(np.average(num_sum), np.average(denom_sum))

# weighted mean 
mu_i = num_sum/denom_sum 
print(np.shape(mu_i), np.average(mu_i))

# sigma_i is (uncertainty in weighted mean) is actually just sqrt(1/denom_sum) from eqn. (2) of the manual 

sigma_i = np.sqrt(1/denom_sum)
print(np.shape(sigma_i), np.average(sigma_i))

# find r_i for target data 
# r_i = f_i^{sci}/mu_i^{ref}

# target data 
target_flux = target[:,2]
#print(target_flux)
target_flux_uncert = target[:,3]
r_i = target_flux/mu_i 

print(np.average(r_i))

# uncertainty propagation - not sure 
sigma_r_i = r_i * np.sqrt ( (target_flux_uncert/target_flux)**2 + (sigma_i/mu_i)**2 )
print(np.average(sigma_r_i))

# write to file -- we need DATE_OBS from the headers of every image, which will require accessing
# header of every .new file (which has the astrometric solutions), but we can do that later, if rlly. need to# for now, just putting the exposure times, but the DATE_OBS from header files will be useful to 
# find images for before and after the expected transit 

# from the email: expected transit ~11:35pm with 24 min. uncertainty in the midtransit, until ~1:25am 

# to get DATE_OBS from header 
date_obs = np.array([])

image_files = np.loadtxt('astrometric_solns.txt', dtype='str')
for ii in range(len(image_files)):
    hdul = fits.open('../TOI_new/{}'.format(image_files[ii]))
    obs = hdul[0].header['DATE-OBS']
    date_obs = np.append(date_obs, obs)

#print(date_obs, np.shape(date_obs)) 

# write to new file 

with open('./analysis_files/{}_{}.dat'.format('deblended_target_star', 'cal' ), 'w+') as file:            

    for DATE_OBS, FLUX, FLUX_UNCERT, WMEAN, WMEAN_UNCERT, RATIO, RATIO_UNCERT in zip(date_obs, target_flux, target_flux_uncert, mu_i,sigma_i, r_i, sigma_r_i):
        file.write('{:20}\t{:20}\t{:20}\t{:20}\t{:20}\t{:20}\t{:20}\n'.format(DATE_OBS,FLUX,FLUX_UNCERT,WMEAN, WMEAN_UNCERT,RATIO, RATIO_UNCERT))
    file.close()


# Plot 
# calibrated target (target_flux/mu_i)
plt.plot(exptime, r_i, '.', label = 'r_i')
plt.legend()
plt.show() 

# check each reference star divided by mu_i as well 
'''
for i in range(len(data_list)):
    data = np.loadtxt('./analysis_files/{}'.format((data_list[i]))) # access data 
    flux = data[:,2] 
    
    plt.plot(exptime, flux/mu_i, '.')
    
    plt.show()
''' 
