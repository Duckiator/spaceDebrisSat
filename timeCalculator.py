import numpy as np

b = 0.0022 #ballistic drag coefficient 
a_i = 450000    #altitude in meters

#i have not figure out when the space debris burns
#i'm assuming it is going to burn at 140000 m
a_f = 140000    #target altitude in meters
g = 9.8 #m/s
r = 0.001016 #radius of Debris in meters
m = np.pi * (r**2) #mass of debris

#Before V terminal

    #centripedal force as net force???

#At V terminal
v_t = -(m*g)/b #Equation to solve for v terminal
                # The negative sign accounts for the direction of the velocity
print(f"V Terminal is: {v_t} m/s")

#Solving for time it takes to reach altitude of 0

t_s = (a_f - a_i)/v_t

t_m = t_s/60 #converting the time from sec to min
t_h = t_m/60 #in hours
t_d = t_h/24 #in days

print(f"The time it takes for space debris to burn {t_d} days")



"""
Figuring out how much force I need to push debris from 800 km to 450 km
"""
v_debris = 17500 #speed of space debris
a_debris = 800000 # altitude of debris in meter
G = 6.674*(10**(-11)) #gravitational constant
M_e = 5.972*(10**24) # Earth mass

KE = 1/2 * m * v_debris #kinetic energy
PE = -G*M_e*m/a_debris  #potential energy
work = -G*M_e*m/a_i - (-G*M_e*m/a_debris)

E_tot = KE + PE + work #total energy needed to move object

print(f"Total energy: {E_tot} J")

Force = E_tot/(a_i - a_debris) #converting energy to force 

print(f"Force Required: {Force} N")

"""
Figuring amount of current needed
"""
#assuming we are using a raspberry pi 
battery = 5 #volts




