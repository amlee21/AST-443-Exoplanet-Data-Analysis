# 11/8/21 running for toi 1636.01 

import numpy as np  
import matplotlib.pyplot as plt 
from datetime import datetime, timedelta, timezone 

# Now just for TOI 1636.01 using: 
# https://exofop.ipac.caltech.edu/tess/target.php?id=269892793

planet_name = np.array(['TOI 1636.01 planet'])
star_name = np.array(['TOI 1636.01'])

# R_earth is 0.0091577 R_sun's so take R_earth*0.0091577 to convert to R_sun 
rad_planet = np.array([18.9843793142131*0.0091577]) #Now in Rsun  [(R_Earth uncertainty is +/- 9.064306)]
rad_star = np.array([0.687841]) # R_sun (uncertainty is +/- 0.0515126) 
period = np.array([1.77298765958882]) # days 
prim_trans= np.array([2458792.135458]) # in BJD for primary transit 
transit_duration = np.array([1.83208927525493]) # [hours] 

transit_depth = (rad_planet[0]/rad_star[0])**2
#print(transit_depth[0])

# Range for declination 
'''
dec_min = 21 # [deg]
dec_max = 61 # [deg]

ra_max  = 19 # [hour] i.e. 7PM 
ra_min  = 1 # [hour] i.e. 1AM
''' 
# Initial limitations for selecting candidates by magnitude, declination, right ascension 
'''
good = np.where( (V_mag < 12) & (dec < dec_max) & (dec > dec_min) & (right_asc > 15) & ( (right_asc >ra_max*15) | (right_asc < ra_min * 15) ) )
data_good = data[good] 

# good data now (yields 60 candidates just based on RA, dec, and magnitude limitations): 
star_name = star_name[good]
planet_name = planet_name[good] 
rad_planet = rad_planet[good] # [solar radii]
period = period[good]         # [days]
semi_major = semi_major[good] # [solar radii]
imp_param = imp_param[good]
right_asc = right_asc[good]   # [deg]
dec = dec[good]               # [deg]
V_mag = V_mag[good]
rad_star = rad_star[good]     # [solar radii] 
prim_trans = prim_trans[good]
print(np.shape(data_good)) # print number of candidates left for selection 
print(np.shape(star_name))
print(np.shape(planet_name))

transit_depth = (rad_planet/rad_star)**2
transit_duration = (period/np.pi) * np.arcsin(np.sqrt( (rad_star+rad_planet)**2 - (imp_param*rad_star)**2)/semi_major)
print(transit_duration)
print(prim_trans)
print(V_mag)
''' 

# Primary transit in JD represents when the planet is right in front of star (middle of its transit)
# so, if we keep adding periods (days) to the primary transit (JD) we basically would get times for when the planet is at the midpoint of its transit
 
#planet = np.where(planet_name == 'WASP-48b')

#print(prim_trans[planet])
#print(period[planet], rad_star[planet], rad_planet[planet], imp_param[planet], transit_duration[planet])


dt_Offset = 2400000.5 # offset between JD and MJD (modern julian date), starts at 11-17-1858

''' 
modified_julian_date = prim_trans - dt_Offset
dt_list = [ ]
print(modified_julian_date)

good = np.where(~np.isnan(modified_julian_date))  # only take indices where MDJ is not a nan 
print(good)
print(modified_julian_date[good])

modified_julian_date = modified_julian_date[good] 
''' 

# New York is 4 hours behind UTC time 

# today's julian day, i.e. august 31, 2021 is 2459458 JD (Wikipedia for Julian Day) 
today_JD = 2459458 # JD for august 31, 2021 
# 8/31/21 to october 8th is 38 so, iterate T0 + period until 2459458 

#oct8_JD = today_JD + 38 

nov8_JD = today_JD + 69 

''' 
for i in range(len(modified_julian_date)):
   
    dt = datetime(1858, 11, 17, tzinfo = timezone.utc) + timedelta(modified_julian_date[i])
    #print(dt.date()) # get only the dates 
'''

# --- Now we want to calculate transit times for the planets --- 

# so take the indices where there is a JD in the catalog, as well as transit depth > 0.008, and transit_duration < 3 hours 

date_obs = prim_trans + period # in JD

# transit_Duration  above was in days, so < 0.125 is  < 3 hours
 
#good = np.where(~np.isnan(date_obs) & (transit_depth > 0.008))

# take primary transits only those without nans for their prim_trans + period 
#date_obs = date_obs[good] 

#star_name = star_name[good]
#planet_name = planet_name[good]
#rad_planet = rad_planet[good] # [solar radii]                                                          
#period = period[good]         # [days]                                                                 
#semi_major = semi_major[good] # [solar radii]                                                          
#imp_param = imp_param[good]
#right_asc = right_asc[good]   # [deg]                                                                  
#dec = dec[good]               # [deg]                                                                  
#V_mag = V_mag[good]
#rad_star = rad_star[good]     # [solar radii]                                                          
#prim_trans = prim_trans[good]

#transit_depth = (rad_planet/rad_star)**2
#transit_duration = (period/np.pi) * np.arcsin(np.sqrt( (rad_star+rad_planet)**2 - (imp_param*rad_star)**2)/semi_major)

day_to_observe = [ ] 

# Run through every primary transit and if it is less than the julian day for october 8th, 2021, 
# add until it is 

for i in range(len(date_obs)):
    
    while (date_obs[i] < nov8_JD): 
    
        date_obs[i] = date_obs[i] + period[i]
 
        if date_obs[i] > today_JD:
            day_to_observe.append([date_obs[i], planet_name[i]])

#print(date_obs)

print(day_to_observe) # julian day for transits that we can potentially observe
print(len(day_to_observe))
print(day_to_observe[0][0])

# what we can observe with transit from september to october 9th ? 
can_observe = [ ] # have list of name + UTC time 

count = 0 
for i in range(len(day_to_observe)):
    dt = datetime(1858, 11, 17, tzinfo = timezone.utc) + timedelta(day_to_observe[i][0]-dt_Offset)
    print(dt)
    can_observe.append([day_to_observe[i][1], str(dt), day_to_observe[i][0]]) # can_observe is a list of lists, [[planet name, date]...]
    count += 1 

for i in range(len(can_observe)):
    print(can_observe[i][0], can_observe[i][1], day_to_observe[i][0]) # prints out planet and the time to observe in UTC



#print(dt_list)

# PLOT 
# ---- 

#plt.plot(V_mag, transit_depth, '+')
#plt.show()


# New file with these candidates 
'''
with open('./exoplanet_candidates.txt', 'w+') as file:
    for planet_name, rad_planet, period, semi_major, imp_param, star_name,  right_asc, dec, V_mag, rad_star, prim_trans in zip(planet_name, rad_planet, period, semi_major, imp_param, star_name, right_asc, dec, V_mag, rad_star, prim_trans): 
        file.write('{:15}\t{:15}\t{:15}\t{:15}\t{:15}\t{:15}\t{:15}\t{:15}\t{:15}\t{:15}\n'.format(planet_name, rad_planet, period, semi_major, imp_param, star_name, right_asc, dec, V_mag, rad_star, prim_trans))

file.close() 
''' 
 
