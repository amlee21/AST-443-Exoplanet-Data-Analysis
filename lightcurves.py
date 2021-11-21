import numpy as np 
import matplotlib.pyplot as plt 

# data files: 

'''
./analysis_files/exptime.dat

refstar_1.dat
refstar_2.dat
refstar_3.dat
refstar_4.dat
refstar_5.dat
refstar_6.dat
refstar_7.dat
refstar_8.dat 

target_star.dat 
 
'''

# Data structure

# col 0: RA
# col 1: DEC
# col 2: Flux 
# col 3: Flux uncertainty 

time = np.loadtxt('./../../analysis_files/exptime.dat')

data_files = np.array(['target_star.dat','refstar_1.dat','refstar_2.dat','refstar_3.dat','refstar_4.dat','refstar_5.dat','refstar_6.dat','refstar_7.dat','refstar_8.dat'])

data_ndb_files = np.array(['target_star.dat', 'refstar_ndb_1.dat','refstar_ndb_2.dat','refstar_ndb_3.dat','refstar_ndb_4.dat','refstar_ndb_5.dat','refstar_ndb_6.dat','refstar_ndb_7.dat','refstar_ndb_8.dat'])

check_files = np.array(['check_target_star.dat','check_refstar_1.dat','check_refstar_2.dat','check_refstar_3.dat','check_refstar_4.dat','check_refstar_5.dat','check_refstar_6.dat','check_refstar_7.dat','check_refstar_8.dat'])

# Plot all light-curves but on single plot 


''' 
for i in range(len(data_files)):
    data = np.loadtxt('./analysis_files/{}'.format((data_files[i])))
    plt.plot(time, np.log10(data[:,2]), '.', label = '{}'.format(data_files[i]))
    plt.xlabel('Time [s]')
    plt.ylabel('log(Counts)')
    plt.legend()
plt.show()

'''

# these are supposedly deblended from seongin's cat files 11/1/2021

# but now I want to test if he actually did do the deblending... there is a diff. but no one is responding

deblended_files = np.array(['deblended_target_star.dat', 'deblended_refstar_1.dat', 'deblended_refstar_2.dat', 'deblended_refstar_3.dat', 'deblended_refstar_4.dat', 'deblended_refstar_5.dat', 'deblended_refstar_6.dat', 'deblended_refstar_7.dat', 'deblended_refstar_8.dat'])

# my own after running SExtractor with -DEBLEND_MINCOUNT 0.5 included in the bash script 

db_files = np.array(['db_target_star.dat', 'db_refstar_1.dat', 'db_refstar_2.dat', 'db_refstar_3.dat', 'db_refstar_4.dat', 'db_refstar_5.dat', 'db_refstar_6.dat', 'db_refstar_7.dat', 'db_refstar_8.dat'])

# Plot all light-curves 3x3 

plt.figure(figsize=(14,8))
for i in range(len(data_files)):
        
    data = np.loadtxt('./../../analysis_files/old_refstar_dat/{}'.format((data_files[i])))
    data_deblended = np.loadtxt('./../../analysis_files/{}'.format((deblended_files[i])))
#    mine_data_db = np.loadtxt('./../../analysis_files/{}'.format((db_files[i])))
    data_ndb = np.loadtxt('./../../analysis_files/{}'.format((data_ndb_files[i])))
    
    check_data = np.loadtxt('./{}'.format(check_files[i]))

    plt.subplot(3,3,i+1)
    plt.plot(time,data[:,2], '.', label = 'previous', alpha = 0.7)
    plt.plot(time, data_deblended[:,2], '.', label = 'deblended', alpha = 0.7)
#    plt.plot(time, mine_data_db[:,2], '.', label = 'db') # seongin's and mine produce same thing 
    #plt.plot(time, data_ndb[:,2], '.', label = 'ndb') 
    #plt.plot(time, check_data[:,2], '.', label = 'check, diam10')
    plt.title('{}'.format(data_files[i]))
    plt.xlabel('Time [s]')
    plt.ylabel('Counts')
    plt.legend()

plt.tight_layout() 
plt.savefig('lightcurves+check_backto10.png')
plt.show()                   
    
