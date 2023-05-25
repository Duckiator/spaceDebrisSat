import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.integrate import solve_ivp

RADIUS_EARTH = 6378.14 * 1e3 # [m]
MASS_EARTH = 5.97219e24 # [kg]
GRAVITATIONAL_CONSTANT = 6.6743e-11 # [m^3 kg^-1 s^-2]

GM = GRAVITATIONAL_CONSTANT * MASS_EARTH

def radiusToAltitude(R):
    return R - RADIUS_EARTH

def altitudeToRadius(h):
    return h + RADIUS_EARTH

def polyfn(h):
        a0 = 7.001985e-02
        a1 = -4.336216e-03
        a2 = -5.009831e-03
        a3 = 1.621827e-04
        a4 = -2.471283e-06
        a5 = 1.904383e-08
        a6 = -7.189421e-11
        a7 = 1.060067e-13

        return a0 + a1*h + a2*(h**2) + a3*(h**3) + a4*(h**4) + a5*(h**5) + a6*(h**6) + a7*(h**7)

def rho(R, F10=70):
    """
    rho is the atmospheric density
    
    F10 = 70 leads to the minimum density: use as default
    
    Source:
    https://www.spaceacademy.net.au/watch/debris/atmosmod.htm
    "A MODEL FROM 180 to 500 KM"
    """
    h = radiusToAltitude(R) * 1e-3 # [m] -> [km]   

    
    if h > 179:
        Ap = 0 # per the website
            
        T = 900 + 2.5 * (F10 - 70) + 1.5 * Ap # [K]
        mu = 27 - 0.012 * (h - 200) # requires h to be [km]
        H = T / mu
        
        atm_density = 6e-10 * np.exp( -(h-175)/H ) # [kg/m^3]
    else:
        atm_density = 10**(polyfn(h))
    
    return atm_density

def T_orb(R):
    """
    T is the orbital period
    """
    T = 2*np.pi * np.sqrt(R**3 / GM)
    return T

def v_orb(R):
    """
    v is the orbital velocity
    """
    v = np.sqrt(GM / R)
    return v

def alpha(R, cd, A, m):
    """   
    Parameters
    ----------
    R : float
        Radius from Earth's center [m]
    cd : float
        Drag coefficient [unitless]
    A : float
        Cross-sectional area [m^2]
    m : float
        Mass [kg]

    Returns
    -------
    deceleration : float
        Drag-induced deceleration [m/s]

    """
    deceleration = 1/2 * rho(R) * v_orb(R)**2 * cd * A / m
    return deceleration

def decayModel(t, R, cd, A, m):
    dRdt = - alpha(R, cd, A, m) * T_orb(R) / np.pi
    return [dRdt]

def atmosphericDensityFunctionValidation():

    # Check our atmospheric density function
    fig1, ax1 = plt.subplots(constrained_layout=True)
    fig1.set_size_inches(5,4)
    
    h = np.linspace(200, 500, num=1000) # altitude [km]
    R = altitudeToRadius(h * 1e3) # radius [m]
    
    F10_range = {70:"xkcd:blue", 
                  # 150:"xkcd:green",
                 300:"xkcd:red"}
    
    for F10, color in F10_range.items():
        density = rho(R, F10=F10)
        ax1.plot(density, h, c=color, label=f'F10 = {F10}')
      
    ax1.legend(loc="upper left", framealpha=1)    
    ax1.invert_xaxis()
    ax1.set_xscale("log")
    ax1.set_xlim([1e-9,1e-13])
    ax1.set_ylim([200,500])
    ax1.yaxis.set_major_locator(MultipleLocator(100))
    ax1.yaxis.set_minor_locator(MultipleLocator(10))
    ax1.grid(which="major", ls='-')
    ax1.grid(which="minor", ls=':')
    ax1.set_xlabel("$\\rho$ [kg m$^{-3}$]")
    ax1.set_ylabel("$h$ [km]")
    ax1.set_title("Model atmospheric density vs. height")
    fig1.savefig("atmospheric_density_function_validation.png", dpi=300)
    return

def solveDrag(drag_start_altitude, burnup_altitude, t_end, cd, A, m, npts=10000):
    R0 = altitudeToRadius(drag_start_altitude)
    t_span = [0, t_end]
    t_eval = np.linspace(*t_span, num=npts)
    
    burnUpEvent = lambda t, R, cd, A, m: burnUp(t, R, burnup_altitude)
    burnUpEvent.terminal = True
    burnUpEvent.direction = -1
    
    solution = solve_ivp(decayModel, t_span, [R0], 
                         args=(cd, A, m), t_eval=t_eval, events=burnUpEvent)
    t = solution.t
    R = solution.y.flatten()
    h = radiusToAltitude(R)
    return t, h, solution

def plotDrag(t, h):   
    fig1, ax1 = plt.subplots(constrained_layout=True)
    fig1.set_size_inches(5,4)
    
    tConv = 1/60/60/24 # [s] -> [days]
    hConv = 1e-3
    ax1.plot(t * tConv, h * hConv)
    ax1.set_xlabel("$t$ [days]")
    ax1.set_ylabel("$h$ [km]")
    ax1.set_xmargin(0)
    ax1.set_xlim([0, 360])
    ax1.grid(which="major", ls='-')
    ax1.grid(which="minor", ls=':')
    ax1.set_title("Model Altitude vs. Days")
    fig1.savefig("modelOfDebris.png", dpi=300)
    return

def burnUp(t, R, burnup_altitude):
    return R[0] - altitudeToRadius(burnup_altitude) 

cd = 1.15
rho_debris = 2700 # [kg/m^3] assume debris is aluminum
radius_debris = 0.4 / 2 * 0.0254 # [in] -> [m]
thickness_debris = 2e-3 # [m]
drag_start_altitude = 500e3 # [km] -> [m]
burnup_altitude = 90e3 # [km] -> [m]

A = np.pi * radius_debris**2 # [m^2]
m = A * thickness_debris * rho_debris # [kg]

t, h, solution = solveDrag(drag_start_altitude, burnup_altitude, 400*24*60*60, cd, A, m)
plotDrag(t,h)
atmosphericDensityFunctionValidation()



""" import numpy as np
from scipy.constants import G
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


#The greatest concentration in space debris is in the altitudes between 800 km & 850km 
#Therefore I am going to assume that the Satellite should be at 850 km
h_orbit_start =  850000 #[m]

#Drag is in effect even at 2000km, but the tendency for satellites to be brought down to Earth easily is in the range of 200-600km
#Because I know what the 500km atmosphere density behaves, I chose this at my starting orbit drag
h_orbit_drag =  500000 #[m]

#space debris reentries, with the much lower velocity of ~ 8 km/s only start to show significant atmospheric interaction below 90 km.
h_orbit_burn =  90000 #[m]

d_in = 0.4 #radius of Debris in meters
r_in = d_in/2
r_m = r_in * 0.0254

aluminum_density=2710 

m = np.pi * (r_m**2) *aluminum_density #mass of debris
drag_area = np.pi * (r_m**2) #Area of which is experiencing drag
c_d = 1.15 #Drag coefficient for a Short Cylinder/Disk


M_e = 5.972e24 # Earth mass [kg]
mu = M_e * G #Standard Gravitational Parameter
r_e = 6.378e6 #Earth Radius [m]
g = 9.81 #Gravity Acceleration [m/s^2]
R = 8.31 #Ideal Gas Constant [J/(mol*K)]

r_orbit_start =  h_orbit_start + r_e
r_orbit_drag =  h_orbit_drag + r_e
r_orbit_burn =  h_orbit_burn + r_e





 #To Calculate Total Energy
v_orbit = np.sqrt(mu/r_orbit_start)
KE = 1/2 * m * v_orbit #kinetic energy
PE = -G*M_e*m/r_orbit_start #potential energy
E_tot = KE + PE 

#Density Function for 180 < altitude (km) < 500
def densityFunction(altitude):

    F10 = 70 #leads to lower bound density
    Ap = 0 #per the website

    T = 900 + 2.5*(F10 - 70) + 1.5*Ap # [Kelvin]
    μ = 27 - 0.012*(altitude - 200) #180 < altitude (km) < 500
    H = T / μ # [km]
    density = 6*10**(-10)*np.exp(-1*(altitude - 175) / H ) #[kg m^(-3)] density
    return density

def velocityFunction(radius):
    PE_C = -G*M_e*m/radius #Changing Potential energy
    work = -G*M_e*m/radius - (-G*M_e*m/r_orbit_start)
    v = np.sqrt(2*(E_tot - PE_C - work)/m) #needs work
    return v 
def velocityFunction(radius):
    v = np.sqrt(mu/radius)
    return v

def orbitalDecayFunction(altitude):
    drag_deceleration = 1/2 * densityFunction(altitude) * (velocityFunction(altitude + r_e)**2)*c_d*drag_area/m
    return drag_deceleration


def ode(t,x):
    dxdt = 
    return [dxdt]

t_span = [0,5]
t_eval = np.linspace(t_span[0], t_span[1], num=1000)
x0 = [1]

soln = solve_ivp(ode, t_span, x0, t_eval=t_eval)

fig, ax = plt.subplots()

ax.plot(soln.t, soln.y.flatten(), c='k', ls='-', label='Numerical solution')

# ax.legend()
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_xmargin(0)
ax.grid(axis='both')
 """

""" 
H_tp = 6.3e3 #Height Scale of the Exponential Fall for Density [m]
temp0_pause = 220 #Intial Temperature at Tropopause [K]
temp0_sphere = 288 #Intial Temperature at Troposphere [K]
M = 0.0289652 #Molar Mass of dry air [kg/mol]
"""


"""
U = 20 #Height between Troposphere and Tropopause
p0 = 1.255 #Standard Atmosphere Intial Density [kg/m^3]
Pa_0 = 101.325 #Standard Atmosphere Intial Pressure [kPa]

h_tropopause = 60000 * 0.3048 #Altitude of Tropopause and converting ft to m [m]

#Link to specific heat capacity
#https://www.researchgate.net/publication/282983416_Finite_element_analysis_of_space_debris_removal_by_high-power_lasers
#I am unsure about it
Cp = 0.52 #Specific Heat Capacity for a 5cm by 5cm space debris

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
def density_func(height):
    altitude = height - r_e
    l = lapseRate_func(altitude)
    if (height>=h_tropopause):
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

for i in range(0,600000):
    plt.dragForce_func(i)

"""

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




