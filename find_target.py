# A. LEE

# find target star 

# 11/2/21 use deblended instead 
# 11/8/21 use minimum distance to target star instead 

import numpy as np 
import matplotlib.pyplot as plt 

#filelist = np.loadtxt('log.txt', dtype=str)
filelist_db = np.loadtxt('deblended_log.txt', dtype = str)
# flux of target stars

flux_target = np.array([])
flux_uncert = np.array([])
ra_target = np.array([])
dec_target = np.array([]) 


# using my own db cat files
flux_target_db = np.array([])
flux_uncert_db = np.array([])
ra_target_db = np.array([])
dec_target_db = np.array([]) 


count = 0 

# RA: 2:57:10.31 hr:min:s => 44.29295833333333 [deg] 

target_ra = (2*15)+(15*(57/60))+(15*10.31/3600) # [deg] 
print(target_ra)
target_dec = 33.31266

for i in range(len(filelist_db)):
    
    #data = np.loadtxt(filelist[i])
    
    data_db = np.loadtxt(filelist_db[i])

    '''
    ra = data[:,3] # [deg]
    dec = data[:,4]
    flux = data[:,5] 
    flux_err = data[:,6] 
    '''
    target_ra = 44.29296 # rounding off to fifth decimal place 
    target_dec = 33.31266 
    #diff = np.abs(ra - target_ra)


    ra_db = data_db[:,3] # [deg]
    dec_db = data_db[:,4]
    flux_db = data_db[:,5] 
    flux_err_db = data_db[:,6] 

    #diff_db = np.sqrt(ra_db - target_ra)
    # distance formula in RA and DEC plane: 
    # dist ~ np.sqrt( ((RA_1 - RA_2)*cos(DEC_1))^2 + (DEC_1 - DEC_2)^2 )
    diff_db = np.sqrt( ( (ra_db - target_ra)*np.cos(dec_db))**2 + (dec_db - target_dec)**2 )

    # index for minimum difference (i.e. index for distance closest to target star) 
    # use this index to access all other values for the target 
    '''
    target = np.argmin(diff) 
        
    target_RA = ra[target] # [deg] 
    target_DEC = dec[target] # [deg] 
    target_flux = flux[target]
    target_flux_uncert = flux_err[target]

    flux_target = np.append(flux_target, target_flux)
    flux_uncert = np.append(flux_uncert, target_flux_uncert)

    ra_target = np.append(ra_target, target_RA)
    dec_target = np.append(dec_target, target_DEC)
    ''' 

    # DB 

    target_db = np.argmin(diff_db) 
        
    target_RA_db = ra_db[target_db] # [deg] 
    target_DEC_db = dec_db[target_db] # [deg] 
    target_flux_db = flux_db[target_db]
    target_flux_uncert_db = flux_err_db[target_db]

    flux_target_db = np.append(flux_target_db, target_flux_db)
    flux_uncert_db = np.append(flux_uncert_db, target_flux_uncert_db)

    ra_target_db = np.append(ra_target_db, target_RA_db)
    dec_target_db = np.append(dec_target_db, target_DEC_db)

    
#print(count)

#print(flux_target)
#print(np.shape(flux_target)) 

print(np.shape(flux_target_db))

'''
with open('./analysis_files/{}.dat'.format('target_star'), 'w+') as file: 
    for RA, DEC, FLUX, FLUX_UNCERT in zip(ra_target, dec_target, flux_target, flux_uncert):
        file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,FLUX,FLUX_UNCERT))
    file.close()
''' 

with open('./analysis_files/{}.dat'.format('deblended_target_star'), 'w+') as file: 
    for RA, DEC, FLUX, FLUX_UNCERT in zip(ra_target_db, dec_target_db, flux_target_db, flux_uncert_db):
        file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,FLUX,FLUX_UNCERT))
    file.close()

exptime = np.loadtxt('./analysis_files/exptime.dat')

plt.plot(exptime, flux_target_db)
plt.show() 
