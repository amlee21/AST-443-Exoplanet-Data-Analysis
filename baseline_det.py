# A. Lee 2021-10-31

# edit -- 11/8/21 see cal.py 
#      -- 11/9/21 replacing exptime with SAH's caltime (hr:m:s from DATE-OBS converted to seconds)
#      -- 11/10/21 try instead normalize just by average flux in flat region from ~ 340 to 370 minutes 
#      -- 11/18/21 binning by time 


# Baseline determination 


import numpy as np 
import matplotlib.pyplot as plt 
from astropy.io import fits 

''' 
Data structure
==============

in target_star_cal.dat

col 0 = date_obs
col 1 = target flux 
col 2 = target flux uncertainty 
col 3 = weighted mean from ref stars (mu_i)
col 4 = uncertainty in weighted mean (sigma_i)
col 5 = ratio (target_flux/mu_i) (r_i)
col 6 = uncertainty in ratio (sigma_r_i) for which I used error propagation... 
''' 

date_obs = np.loadtxt('./analysis_files/deblended_target_star_cal.dat', usecols = 0, dtype = 'str')
#print(date_obs)

exptime = np.loadtxt('./analysis_files/exptime.dat')

# Identify images that were taken before and after the expected transit
# identified by hand:                                                                                        
# start: expected ~ 11:35pm 2021-10-02                                                                       
# end  : expected ~ 01:25am 2021-10-03                                                                       
# just before expected transit: DATE-OBS = '2021-10-03T03:34:17.216'                                         
# just after                  : DATE-OBS = '2021-10-03T05:26:43.963'       

# avoiding column with date_obs since that is a string and so read it in separately 
data = np.loadtxt('./analysis_files/deblended_target_star_cal.dat', usecols = (1,2,3,4,5,6))

flux        = data[:,0]
print(np.average(flux))
flux_uncert = data[:,1]
mu_i        = data[:,2]
sigma_i     = data[:,3]
r_i         = data[:,4]
sigma_r_i   = data[:,5]

before = np.where(date_obs == '2021-10-03T03:34:17.216')
after  = np.where(date_obs == '2021-10-03T05:26:43.963')

# np.where returns array([indices]) so to get the one and only then do 
# array[0][0] --> array[0] returns [indices] and then array[0][0] gives the first index of the [indices] 

#print(before[0][0]) # index for image before 
#print(after[0][0])  # index for image after 

ind_just_before = before[0][0]
ind_just_after  = after[0][0]

#print(date_obs[ind_before])
#print(date_obs[ind_after])

#date_obs_before = date_obs[:ind_before+1] # 02:01:59.528 to 03:34:17.216
#date_obs_after  = date_obs[ind_after:]    # 05:26:43.963 to 06:51:06.636

#print(date_obs_before[0], date_obs_before[-1]) 
#print(date_obs_after[0], date_obs_after[-1])

before_flux = flux[:ind_just_before+1]
after_flux = flux[ind_just_after:]

# ===========================================================================================================
# for time x-axis in seconds (converting hr:m:s from DATE-OBS to seconds) 
# determination hr:mm:ss.s from DATE-OBS


#loading text file which as file name and exposure time.
f=np.loadtxt('DATE-OBS.txt', dtype=str)

#take only the file names or first coloumn
c=np.array([f[:,1]])

#it's a 2D array, so we convert it to 1D for analysis
s=c.flatten()

#print(s.shape)
#print(s)

#Now we split the text in the file in strings in a list w 
#the end goal is get the number, we dont want the T so we get rid of it
w=np.array([])
for i in s:
    y=i.split('T')
    w=np.append(w,y)

#prints out shape of the w array now it has date and time

print(w.shape)

#However it is 1D so we need to convert to 2D to retrive time
n=np.reshape(w, (319,2))

#saving hour in p
p=n[:,1]

#now we convert the string from the array in hh:mm:ss.sss to just ss.sss
import datetime

caltime=np.array([])
for i in range(len(p)):
    test=p[i]
    h,m,s=test.split(':')
    total_sec=(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
    caltime=np.append(caltime,total_sec/60)
    
#print(cal_time)

print(np.shape(before_flux), np.shape(after_flux), np.shape(flux))


# TRY SOME PLOTS 


# Plot for directly before and after the expected ingress/egress times 
plt.subplot(2,1,1)
plt.title('Identify images before and after transit')
plt.plot(caltime, flux, '.', label = 'full flux')
plt.plot(caltime[:ind_just_before+1], before_flux, '.', label = 'just before expected transit')
plt.plot(caltime[ind_just_after:], after_flux, '.', label = 'just after')



plt.legend()
plt.show()

#=== try ~ten minutes before and after expected == 

before = np.where(date_obs == '2021-10-03T03:26:00.099')
after  = np.where(date_obs == '2021-10-03T05:35:00.252')

ind_before = before[0][0]
ind_after  = after[0][0]

print(date_obs[ind_before])
print(date_obs[ind_after])

date_obs_before = date_obs[:ind_before+1] # 02:01:59.528 to 03:34:17.216
date_obs_after  = date_obs[ind_after:]    # 05:26:43.963 to 06:51:06.636

before_flux = flux[:ind_before+1]
after_flux = flux[ind_after:]

# Plot for ten minutes before and after the expected ingress/egress times
#plt.subplot(2,1,2)
plt.plot(caltime, flux, '.', color = 'grey', label = 'full flux')
plt.plot(caltime[:ind_before+1], before_flux, '.', label = '~10 min before expected transit', alpha = 0.7)
plt.plot(caltime[ind_after:], after_flux, '.', label = '~10 min after expected end of transit', alpha = 0.7)
plt.title('Raw Target Flux')
plt.xlabel('DATE-OBS converted to minutes')
plt.ylabel('Counts')
# Flat part of the raw target flux using time
good = np.where( (caltime > 340) & (caltime < 370) )

plt.plot(caltime[good], flux[good], 'g', label = 'region to calculate baseline flux')

plt.legend()
plt.show() 

# Calibrated target flux normalized by their respective baselines before and after the "expected" transit 
cal_before_flux = before_flux/np.average(before_flux)
cal_after_flux = after_flux/np.average(after_flux) 

# Plot - compare the uncalibrated and "calibrated" for baseline det.
'''
plt.subplot(3,1,1)
plt.plot(exptime[:ind_before+1], before_flux, '.', label = '~10 min before expected transit')
plt.plot(exptime[ind_after:], after_flux, '.', label = '~10 min after expected end of transit')
plt.legend()
plt.subplot(3,1,2)
plt.plot(exptime[:ind_before+1], cal_before_flux, '.', label = 'baseline ~10 min before')
plt.plot(exptime[ind_after:], cal_after_flux, '.', label = 'baseline ~10 min after')
plt.legend()
plt.subplot(3,1,3)
plt.plot(exptime, flux, '.', label = 'full')
plt.legend()
plt.show()
''' 

# better to divide by the flat part of the raw target flux *after* the expected transit 

cal_r_i       = r_i/np.average(flux[good])

#cal_r_i       = r_i/np.average(before_flux)

cal_sigma_r_i = sigma_r_i/np.average(flux[good])

avg_cal_r_i = np.average(cal_r_i)

print(np.round(avg_cal_r_i, decimals = 2))

# Plot normalized r_i (normalized by baseline flux *after* expected transit) and flux 
'''
plt.figure(figsize=(14,8))
plt.subplot(3,1,1)
plt.plot(caltime, (r_i), '.', label = 'r_i')
plt.legend()

plt.subplot(3,1,2)
plt.plot(caltime, (cal_r_i), '.', label = 'r_i divided by baseline flux after')
plt.plot(caltime[:ind_just_before+1], cal_r_i[:ind_just_before+1], '.',  label = 'just before transit')
plt.plot(caltime[ind_just_after:], cal_r_i[ind_just_after:], '.', label = 'just after transit')
plt.ylim(0.85,1.05)
plt.legend(fontsize='x-small') 

plt.subplot(3,1,3)
plt.plot(caltime, (cal_r_i), '.', label = 'r_i divided by baseline flux after')
plt.plot(caltime[:ind_before+1], cal_r_i[:ind_before+1], '.',  label = '10 min before transit')
plt.plot(caltime[ind_after:], cal_r_i[ind_after:], '.', label = '10 min after transit')
plt.ylim(0.85,1.05)
#plt.xlim(6000,8000)

plt.legend(fontsize='x-small') 

#plt.savefig('div_baseline_4.5.png')
plt.show() 
''' 

print(np.average(r_i))
print(np.average(flux))
print(np.average(after_flux))


plt.figure(figsize = (10,7))
#plt.subplots_adjust(right=0.2)
#plt.subplot(3,1,1)
#plt.subplot(2,1,1)

plt.subplot(3,1,1)
plt.plot(caltime, flux, '.', color = 'grey', markersize = 2.5, label = 'uncal. tflux')
plt.errorbar(caltime, flux, yerr = flux_uncert, fmt = '.', color = 'grey', markersize = 2.5)
plt.plot(caltime[good], flux[good], 'go', label = 'region used to calculate baseline flux')

plt.axvline(x = caltime[ind_just_before], color = 'red', linestyle = 'dashed', label = '{} UTC'.format(date_obs[ind_just_before]))
plt.axvline(x = caltime[ind_just_after], color = 'blue', linestyle = 'dashed', label = '{} UTC'.format(date_obs[ind_just_after]))

plt.axvline(x = caltime[ind_before], color = 'red', linestyle = 'dashed', label = '{} UTC'.format(date_obs[ind_before]), alpha = 0.5)
plt.axvline(x = caltime[ind_after], color = 'blue', linestyle = 'dashed', label = '{} UTC'.format(date_obs[ind_after]), alpha = 0.5)
plt.axvline(x = caltime[0], color = 'magenta', linestyle = 'dashed', label = 'first exp. {} UTC'.format(date_obs[0]), alpha = 0.5)

plt.ylabel('Counts')
plt.xlim(caltime[0]-5,caltime[-1]+5)
plt.legend(bbox_to_anchor = (1.0,1.04), loc='upper left')
#plt.tight_layout(rect=[0,0,0.75,1])
#plt.tight_layout(rect=[0,0,0.5,5])
# --

plt.subplot(3,1,2)
plt.plot(caltime, r_i, '.', color = 'grey', markersize = 2.5, label = 'cal.tflux, r_i')
plt.errorbar(caltime, r_i, yerr = sigma_r_i, color = 'grey', fmt = '.', markersize = 2.5)

plt.axvline(x = caltime[ind_just_before], color = 'red', linestyle = 'dashed')
plt.axvline(x = caltime[ind_just_after], color = 'blue', linestyle = 'dashed')

plt.axvline(x = caltime[ind_before], color = 'red', linestyle = 'dashed', alpha = 0.5)
plt.axvline(x = caltime[ind_after], color = 'blue', linestyle = 'dashed', alpha = 0.5)
plt.axvline(x = caltime[0], color = 'magenta', linestyle = 'dashed', alpha = 0.5)

#plt.axhline(y = 1.0, color = 'orange', linestyle = 'dashed', alpha = 0.5)

#plt.ylim(0.85,1.10)

plt.ylabel('Relative Flux')
plt.xlim(caltime[0]-5,caltime[-1]+5)
plt.legend(bbox_to_anchor = (1.04,1), loc='upper left')
#plt.tight_layout(rect=[0,0,0.75,1])

#plt.show()
#-- 

# Bin the datapoints using time; e.g. every 11 minutes 
    
binned_flux = np.array([])
binned_uncert = np.array([]) 

N = 11 # number of minutes binning by 

beginning = caltime[0] # time of first exposure 
ending = beginning + N # endtime of first binning 

# keep running loop until "ending" is greater than the time of the last exposure 

plt.subplot(3,1,3)
while (ending < caltime[-1]):
    # find indices for datapoints that fall within the time binning (every eleven minutes) 
    bin = np.where( (caltime > beginning) & (caltime < ending) )

    bintime = caltime[bin]
    print(np.shape(bintime)) # check how many images are being binned in each set 
    
    # bin them 
    bin_flux = np.average(cal_r_i[bin])
    # uncertainty is: (standard deviation in fluxes/ sqrt(N)) where N is the number of measurements 
    # the number of measurements per each set exposures being binned could be different
    # since we are binning every image that falls within each N minute time interval

    bin_uncert = np.std(cal_r_i[bin]) / np.sqrt(np.size(bin))
    
    # append them to new arrays 
    binned_flux = np.append(binned_flux, bin_flux)
    binned_uncert = np.append(binned_uncert, bin_uncert) 

    beginning = beginning + N + 1 # start of next binning
    ending = ending + N + 1 # end of next beginning 
    
    # if there in a binning, plot binned point
    if (np.size(bin) > 0):
        plt.plot(ending, bin_flux/avg_cal_r_i, 'k.', label = 'bin every {} min.'.format(N))
        plt.errorbar(ending, bin_flux/avg_cal_r_i, yerr = bin_uncert/avg_cal_r_i, fmt = 'k.')

# Binned Light Curve: 

plt.plot(caltime, (cal_r_i/avg_cal_r_i), '.', color = 'grey', markersize = 2.5,  label = 'cal. and norm. tflux')
#plt.errorbar(caltime, (cal_r_i/avg_cal_r_i), yerr = sigma_r_i/avg_cal_r_i, fmt = '.', color = 'grey', markersize = 1.5)
plt.axvline(x = caltime[ind_just_before], color = 'red', linestyle = 'dashed')
plt.axvline(x = caltime[ind_just_after], color = 'blue', linestyle = 'dashed')

plt.axvline(x = caltime[ind_before], color = 'red', linestyle = 'dashed', alpha = 0.5)
plt.axvline(x = caltime[ind_after], color = 'blue', linestyle = 'dashed', alpha = 0.5)
plt.axvline(x = caltime[0], color = 'magenta', linestyle = 'dashed', alpha = 0.5)

#plt.axhline(y = 1.0, color = 'orange', linestyle = 'dashed', alpha = 0.5)

#plt.axhline(y=1.01, color = 'green', alpha = 0, label = 'estimated baseline') 


plt.ylim(0.95,1.05)
plt.ylabel('Relative Flux')
plt.xlabel('DATE-OBS (hr:m:s) converted to [mins]')

# Get rid of duplicate labels in legend 
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor = (1.04,1), loc='upper left')

plt.tight_layout()
plt.xlim(caltime[0]-5,caltime[-1]+5)
plt.show()

    




