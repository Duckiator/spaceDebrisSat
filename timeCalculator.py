import numpy as np
from scipy.constants import G


M_e = 5.972e24 # Earth mass [kg]
mu = M_e * G #Standard Gravitational Parameter
r_e = 6.378e6 #Earth Radius [m]
g = 9.81 #Gravity Acceleration [m/s^2]
igc = 8.31 #Ideal Gas Constant [J/(mol*K)]

c_d = 1.15 #Drag coefficient for a Short Cylinder/Disk

H_tp = 6.3e3 #Height Scale of the Exponential Fall for Density [m]
temp0 = 220 #Intial Temperature at Tropopause
mm = 0.0289652 #Molar Mass of dry air [kg/mol]

l = temp/altitude #Lapse rate
u = 20 #Height between Troposphere and Tropopause
p0 = 1.255 #Standard Atmosphere

height = r_e + altitude #the height of where the space debris from the center of the Earth
p = p0(1-(l*u/temp0))**(g*mm/(igc*l)-1)*np.e**(-(height-u/H_tp))




orbit_start =  
orbit_drag = 
orbit_burn =









 #ballistic drag coefficient 
a_i = 450000    #altitude in meters

#i have not figure out when the space debris burns
#i'm assuming it is going to burn at 140000 m
a_f = 140000    #target altitude in meters
g = 9.8 #m/s
d_in = 0.4 #radius of Debris in meters
r_in = d_in/2

r_m = r_in * 0.0254

m = np.pi * (r_m**2) #mass of debris

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




