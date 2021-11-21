# A. LEE 

# 11/2/21 edit -- using deblended exposures instead 
# 11/8/21 edit -- use minimum distance to find ref stars instead, not just RA 

import numpy as np 

#filelist = np.loadtxt('log.txt', dtype=str)
deblended_filelist = np.loadtxt('deblended_log.txt', dtype = str)

count = 0 

# ra of stars that are close to target 
# ====================================
ref_ra = np.array([44.3404061, 44.4029161, 44.4315991, 44.2857892,44.1930316, 44.3772505, 44.2431082
, 44.2634690]) # [deg] 

# dec of stars close to target
# ============================ 
ref_dec = np.array([33.33687, 33.30928, 33.31817, 33.30154, 33.288915, 33.271813, 33.43593, 33.42910])

# For the target 
# ==============
# RA: 2:57:10.31 hr:min:s => 44.29295833333333 [deg] 

# time (s) for each image:

target_ra = (2*15)+(15*(57/60))+(15*10.31/3600) # [deg] 
print(target_ra)

# run through every ref star's ra 
for jj in range(len(ref_ra)):

    # initialize arrays empty for storage 
    flux_ref = np.array([])
    flux_uncert = np.array([])
    ra_ref = np.array([])
    dec_ref = np.array([])

    # initialize arrays at empty for storage of my own deblended cat files. 11/2/21
    flux_ref_db = np.array([])
    flux_uncert_db = np.array([])
    ra_ref_db = np.array([])
    dec_ref_db = np.array([]) 

    # now for every ref star ra we have, let's run through every cat file (image) 
    # and locate the ref star in each image and its data

    for i in range(len(deblended_filelist)):

        #data = np.loadtxt(filelist[i])
        data_db = np.loadtxt(deblended_filelist[i]) # deblended data 
        
        '''
        ra = data[:,3] # [deg]
        dec = data[:,4]
        flux = data[:,5] 
        flux_err = data[:,6] 
        ''' 
        reference_ra = ref_ra[jj] # RA of reference stars 
        reference_dec = ref_dec[jj] 
        #diff = np.abs(ra - reference)
        
        # DB 
        ra_db = data_db[:,3]
        dec_db = data_db[:,4]
        flux_db = data_db[:,5]
        flux_err_db = data_db[:,6] 

        #diff_db = np.abs(ra_db - reference) 
        
        diff_db = np.sqrt( ( (ra_db - reference_ra)*np.cos(dec_db))**2 + (dec_db - reference_dec)**2 )
    # index for minimum difference (i.e. index for minimum distance from ref stars)
    # use this index to access all other values for the reference stars 

        #ref = np.argmin(diff) 
        ref_db = np.argmin(diff_db) 
        
        '''
        ref_RA = ra[ref] # [deg] 
        ref_DEC = dec[ref] # [deg] 
        ref_flux = flux[ref]
        ref_flux_uncert = flux_err[ref]

        flux_ref = np.append(flux_ref, ref_flux)
        flux_uncert = np.append(flux_uncert, ref_flux_uncert)

        ra_ref = np.append(ra_ref, ref_RA)
        dec_ref = np.append(dec_ref, ref_DEC)
        ''' 

        # DB 
        ref_RA_db = ra_db[ref_db]
        ref_DEC_db = dec_db[ref_db]
        ref_flux_db = flux_db[ref_db]
        ref_flux_uncert_db = flux_err_db[ref_db] 

        flux_ref_db = np.append(flux_ref_db, ref_flux_db)
        flux_uncert_db = np.append(flux_uncert_db, ref_flux_uncert_db)
        
        ra_ref_db = np.append(ra_ref_db, ref_RA_db)
        dec_ref_db = np.append(dec_ref_db, ref_DEC_db)
        
    #print(jj+1, np.shape(ra_ref), np.shape(dec_ref), np.shape(flux_ref), np.shape(flux_uncert))
    print(jj+1, np.shape(ra_ref_db), np.shape(dec_ref_db), np.shape(flux_ref_db), np.shape(flux_uncert_db))

    # write to file, e.g. one row of values for every image 
    
    with open('./analysis_files/{}_{}.dat'.format('deblended_refstar', jj+1), 'w+') as file:            

        for RA, DEC, FLUX, FLUX_UNCERT in zip(ra_ref_db, dec_ref_db, flux_ref_db, flux_uncert_db):
            file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,FLUX,FLUX_UNCERT))
        file.close() 
    
    '''
    with open('./analysis_files/{}_{}.dat'.format('db_refstar', jj+1), 'w+') as file:
        
        for RA, DEC, FLUX, FLUX_UNCERT in zip(ra_ref_db, dec_ref_db, flux_ref_db, flux_uncert_db):
            file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,FLUX,FLUX_UNCERT))
        file.close() 

    with open('./analysis_files/{}_{}.dat'.format('refstar_ndb', jj+1), 'w+') as file:            

        for RA, DEC, FLUX, FLUX_UNCERT in zip(ra_ref, dec_ref, flux_ref, flux_uncert):
            file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,FLUX,FLUX_UNCERT))
        file.close() 
    ''' 
        
'''
with open('./analysis_files/{}.dat'.format('target_star'), 'w+') as file: 
    for RA, DEC, FLUX, FLUX_UNCERT in zip(ra_target, dec_target, flux_target, flux_uncert):
        file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,FLUX,FLUX_UNCERT))
    file.close()
''' 
