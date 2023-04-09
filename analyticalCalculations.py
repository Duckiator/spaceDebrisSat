import numpy as np
from scipy.constants import G


#The greatest concentration in space debris is in the altitudes between 800 km & 850km 
#Therefore I am going to assume that the Satellite should be at 850 km
orbit_start =  850000 #[m]
#Drag is in effect even at 2000km, but the tendency for satellites to be brought down to Earth easily is in the range of 200-600km
#Therefore I assume that 600km would be perfect for micro debris to experience drag to be brought to earth
orbit_drag =  600000 #[m]

#Probably has to deal with Atmospheric Entry Heat math
orbit_burn =  

d_in = 0.4 #radius of Debris in meters
r_in = d_in/2
r_m = r_in * 0.0254
m = np.pi * (r_m**2) #mass of debris
drag_area = np.pi * (r_m**2) #Area of which is experiencing drag
c_d = 1.15 #Drag coefficient for a Short Cylinder/Disk


M_e = 5.972e24 # Earth mass [kg]
mu = M_e * G #Standard Gravitational Parameter
r_e = 6.378e6 #Earth Radius [m]
g = 9.81 #Gravity Acceleration [m/s^2]
R = 8.31 #Ideal Gas Constant [J/(mol*K)]



H_tp = 6.3e3 #Height Scale of the Exponential Fall for Density [m]
temp0_pause = 220 #Intial Temperature at Tropopause [K]
temp0_sphere = 288 #Intial Temperature at Troposphere [K]
M = 0.0289652 #Molar Mass of dry air [kg/mol]


U = 20 #Height between Troposphere and Tropopause
p0 = 1.255 #Standard Atmosphere Intial Density [kg/m^3]
Pa_0 = 101.325 #Standard Atmosphere Intial Pressure [kPa]

h_tropopause = 60000 * 0.3048 #Altitude of Tropopause and converting ft to m [m]

#https://www.researchgate.net/publication/282983416_Finite_element_analysis_of_space_debris_removal_by_high-power_lasers
#Link to specific heat capacity
#I am unsure about it
Cp = 0.52 #Specific Heat Capacity for a 5cm by 5cm space debris


height = r_e +  #the height of where the space debris from the center of the Earth

#function to help determine what the temp is at a certain altitude
#https://en.wikipedia.org/wiki/Lapse_rate
#I got stuck on how to address the changing temperature, I probably can add another parameter for it
def lapseRate_func(altitude):
    if (altitude>=h_tropopause):
        l = g/Cp #Dry adiabatic Lapse rate 
    else:
        #I haven't defined any of the variables for this equation yet because I got stuck
        l = g*((1+H_v*r/(R_sd*temp))/(Cpd+H_v**2*r/(R_sw*temp**2)))
    return l


#The function figure out the air density for the certain altitude
def density_func(altitude):
    l = lapseRate_func(altitude)
    if (altitude>=h_tropopause):
        p = p0*(1-(l*U/temp0_pause))**(g*M/(R*l)-1)*np.e**(-(height-U/H_tp))
    else:
        p = Pa_0*M/(R*temp0_sphere)*(1-(l*U/temp0_sphere))**(g*M/(R*l)-1)
    return p
#The function determines the drag force applied to the space debris
def dragForce_func(altitude):
    height = altitude + r_e #Figuring height from the center of the Earth
    v = np.sqrt(mu/height)
    dragForce = 0.5*density_func(altitude)*c_d*drag_area*v**2
    return dragForce

"""

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




Figuring out how much force I need to push debris from 800 km to 450 km

v_debris = 17500 #speed of space debris
a_debris = 800000 # altitude of debris in meter


KE = 1/2 * m * v_debris #kinetic energy
PE = -G*M_e*m/a_debris  #potential energy
work = -G*M_e*m/a_i - (-G*M_e*m/a_debris)

E_tot = KE + PE + work #total energy needed to move object

print(f"Total energy: {E_tot} J")

Force = E_tot/(a_i - a_debris) #converting energy to force 

print(f"Force Required: {Force} N")


Figuring amount of current needed

#assuming we are using a raspberry pi 
battery = 5 #volts

"""




