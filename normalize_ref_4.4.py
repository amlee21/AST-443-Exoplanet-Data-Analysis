# 11/8/21 remove 8th reference star because bottom jump whereas every other ref. star becomes flat 

import numpy as np 
import matplotlib.pyplot as plt 

deblended_ref = np.loadtxt('./analysis_files/deblended_ref.txt', dtype = 'str')
exptime = np.loadtxt('./analysis_files/exptime.dat')
caltime = np.loadtxt('./caltime.txt') # [s]
# we need to exclude ref stars 1, 2, and 5 -- and it is possible that our RA is wrong for them 

for ii in range(len(deblended_ref)):
    ref_data = np.loadtxt('./analysis_files/{}'.format(deblended_ref[ii]))
    ref_flux = ref_data[:,2] 
    
    #print('file: {}'.format(deblended_ref[ii]), ref_flux/np.average(ref_flux))
    
    bad = np.where( (ref_flux < 0.5*np.amax(ref_flux)) )
    
    ref_flux_copy = ref_flux.copy() 
    ref_flux[bad] = np.nan 
    
    print('file: {}'.format(deblended_ref[ii]), ref_flux_copy/np.nanmean(ref_flux))

    plt.subplot(3,3,ii+1)
    plt.title('{}'.format(deblended_ref[ii]))
    plt.plot(exptime, ref_flux_copy/np.nanmean(ref_flux), '.')

plt.tight_layout()
plt.show() 

#normalized_flux = np.array([])
#normalized_uncert = np.array([])

plt.figure(figsize = (14,7))
for ii in range(len(deblended_ref)):
    ref_data = np.loadtxt('./analysis_files/{}'.format(deblended_ref[ii]))
    ref_flux = ref_data[:,2] 
    
    #print('file: {}'.format(deblended_ref[ii]), ref_flux/np.average(ref_flux))
    
    bad = np.where( (ref_flux < 0.5*np.amax(ref_flux)) )
    
    ref_flux_copy = ref_flux.copy() 
    ref_flux[bad] = np.nan 
    
    #print('file: {}'.format(deblended_ref[ii]), ref_flux_copy/np.nanmean(ref_flux))

    normalized_flux = ref_flux_copy/np.nanmean(ref_flux)
    normalized_uncert = ref_data[:,3]/np.nanmean(ref_flux)

    plt.plot(caltime/60, normalized_flux, '.', label = '{}'.format(deblended_ref[ii]))
        
    print(ii+1, np.shape(ref_data[:,0]), np.shape(ref_data[:,1]), np.shape(normalized_flux), np.shape(normalized_uncert))

    with open('./analysis_files/{}_{}.dat'.format('deblended_norm_refstar',ii+1), 'w+') as file:
        for RA, DEC, NORM_FLUX, NORM_UNCERT in zip(ref_data[:,0], ref_data[:,1], normalized_flux, normalized_uncert):
            file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,NORM_FLUX,NORM_UNCERT))
        file.close()


#plt.xlim(0,16000)
plt.legend()
plt.grid()
#plt.tight_layout()
plt.ylabel('Counts')
plt.xlabel('DATE-OBS (hr:m:s) converted to Time [min]')
plt.savefig('normalized_ref_stars_4.4_db_min_dist.png', dpi =300)
plt.show() 

shift = 0 
plt.figure(figsize = (14,7))
for ii in range(len(deblended_ref)):
    ref_data = np.loadtxt('./analysis_files/{}'.format(deblended_ref[ii]))
    ref_flux = ref_data[:,2] 
    
    #print('file: {}'.format(deblended_ref[ii]), ref_flux/np.average(ref_flux))
    
    bad = np.where( (ref_flux < 0.5*np.amax(ref_flux)) )
    
    ref_flux_copy = ref_flux.copy() 
    ref_flux[bad] = np.nan 
    
    #print('file: {}'.format(deblended_ref[ii]), ref_flux_copy/np.nanmean(ref_flux))

    normalized_flux = ref_flux_copy/np.nanmean(ref_flux)
    normalized_uncert = ref_data[:,3]/np.nanmean(ref_flux)

    plt.plot(caltime/60, normalized_flux + shift, '.', label = '{}'.format(deblended_ref[ii]))
        
    print(ii+1, np.shape(ref_data[:,0]), np.shape(ref_data[:,1]), np.shape(normalized_flux), np.shape(normalized_uncert))

    with open('./analysis_files/{}_{}.dat'.format('deblended_norm_refstar',ii+1), 'w+') as file:
        for RA, DEC, NORM_FLUX, NORM_UNCERT in zip(ref_data[:,0], ref_data[:,1], normalized_flux, normalized_uncert):
            file.write('{:20}\t{:20}\t{:20}\t{:20}\n'.format(RA,DEC,NORM_FLUX,NORM_UNCERT))
        file.close()

    shift+=1 

#plt.xlim(0,16000)
plt.legend(fontsize = 'small', bbox_to_anchor = (1.04,1), loc='upper left')
plt.grid()
plt.tight_layout()
plt.ylabel('Normalized Flux')
plt.xlabel('DATE-OBS (hr:m:s) converted to Time [min]')
plt.savefig('normalized_ref_stars_shifts.png', dpi = 300)
plt.title('Normalized Reference Star Light Curves')
plt.show() 
