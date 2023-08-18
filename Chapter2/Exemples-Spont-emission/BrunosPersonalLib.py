#Qutip
from qutip import *
from qutip.piqs import *

#from collections import namedtuple

#Scipy
import scipy as sp
import scipy.integrate as integrate
import scipy.special as special
import scipy.sparse as spr
from scipy.sparse.linalg import eigs
from scipy.integrate import quad, dblquad

#Matplotlib
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#For latex fonts in the plots
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#plt.style.use('ggplot')

# For indexing plots smartly and automatically
import string
alphabet = list(string.ascii_lowercase)

#Math
import csv
import numpy as np
import math as mt
from sympy import Matrix

#For doing an animation of the Bloch sphere
from pylab import *
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Bonus
import itertools as itr
from tqdm import tqdm_notebook
import time
import imageio
import time, sys
from IPython.display import clear_output
from scipy.sparse.linalg import eigs
import pickle


#################################
#    Complex integration
#################################
def complex_quadrature(func, a, b, **kwargs):
    def real_func(x):
        return np.real(func(x))
    def imag_func(x):
        return np.imag(func(x))
    real_integral = quad(real_func, a, b, **kwargs)
    imag_integral = quad(imag_func, a, b, **kwargs)
    return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])

###################################################################################################
#    Animated Poincaré sphere
#
#   This function was adapted from Tanay Roy: 
#      https://sites.google.com/site/tanayroysite/articles/bloch-sphere-animation-using-qutip
#   Thanks Tanay :)
###################################################################################################
def Animate_Poincare(states, animation_name_dot_gif, duration = 0.1, save_all = False): 


    b = Bloch()

    b.vector_color = ['r', 'b']

    b.view = [-40,30]

    images=[]

    try:

        length = len(states)

    except:

        length = 1

        states = [states]

    ## normalize colors to the length of data ##

    nrm = mpl.colors.Normalize(0,length)

    colors = cm.cool(nrm(range(length))) # options: cool, summer, winter, autumn etc.


    ## customize sphere properties ##

    b.point_color = list(colors) # options: 'r', 'g', 'b' etc.

    b.point_marker = ['o']

    b.point_size = [30]
    
    b.xlabel= [r'$\left|H\right>$', r'$\left|V\right>$']
    b.ylabel= [r'$\left|A\right>$', r'$\left|D\right>$']
    b.zlabel= [r'$\left|R\right>$', r'$\left|L\right>$']

    

    for i in range(length):

        b.clear()

        b.add_vectors(states[i])

        #b.add_vectors(states[:(i+1)],'point')

        if save_all:

            b.save(dirc='tmp') #saving images to tmp directory

            filename="tmp/bloch_%01d.png" % i

        else:

            filename='temp_file.png'

            b.save(filename)

        images.append(imageio.imread(filename))

    imageio.mimsave(animation_name_dot_gif, images, duration=duration) 
    
    
    
    
#################################
#    Defining spin operators 
#################################

#Basis vectors

# Spin states
spin_dw = tensor(ket('1'),ket('1'))
spin_up = tensor(ket('1'),ket('0')) 

# Negative trion states
trion_dw = tensor(ket('0'),ket('1')) 
trion_up = tensor(ket('0'),ket('0')) 

# Left and Right lowering and raising operators
σ_R = spin_up*trion_up.dag()
σ_L = spin_dw*trion_dw.dag()

σ_Rd = σ_R.dag()
σ_Ld = σ_L.dag()

# Left and Right Pauli operators
σz_R = trion_up*trion_up.dag() - spin_up*spin_up.dag()
σz_L = trion_dw*trion_dw.dag() - spin_dw*spin_dw.dag()

σx_R = 0.5*(σ_R + σ_Rd)
σx_L = 0.5*(σ_L + σ_Ld)

σy_R = 0.5*(σ_R + σ_Rd)/1j
σy_L = 0.5*(σ_L + σ_Ld)/1j

# Projectors definitions
Π_trion = σ_Rd*σ_R + σ_Ld*σ_L # Projector in the trion subspace - it's average value gives the excitation probability
Π_spin = σ_R*σ_Rd + σ_L*σ_Ld # Projector in the trion subspace - it's average value gives the probability of being in the spin subspace

Coh_spin_subspace = spin_up*spin_dw.dag() + spin_dw*spin_up.dag()
Coh_trion_subspace = trion_up*trion_dw.dag() + trion_dw*trion_up.dag()



#################################
#    Defining the Hamiltonian 
#################################

# Hamiltonian continuous coherent drive
def H(detuning, L_pump_coefficient, R_pump_coefficient, γ = 1.0): 
    
    H_QD = detuning*(σ_Ld*σ_L + σ_Rd*σ_R)
    #H_L = detuning*sigma_Ld*sigma_L - 1j*np.sqrt(gamma)*(np.conj(alpha_in_L)*sigma_L - alpha_in_L*sigma_Ld)
    #H_R = detuning*sigma_Rd*sigma_R - 1j*np.sqrt(gamma)*(np.conj(alpha_in_R)*sigma_R - alpha_in_R*sigma_Rd)
    
    if (type(R_pump_coefficient) is int) or (type(R_pump_coefficient) is float) or (type(R_pump_coefficient) is float64): #For time independent Hamiltonians, i.e. continuous drive
        #print('a')
        H_pump_L = -1j*np.sqrt(γ)*(np.conj(L_pump_coefficient)*σ_L - L_pump_coefficient*σ_Ld)
    
        H_pump_R = -1j*np.sqrt(γ)*(np.conj(R_pump_coefficient)*σ_R - R_pump_coefficient*σ_Rd)
        
        return H_QD + H_pump_L + H_pump_R
    
    else: #For time dependent Hamiltonians, i.e. when considering pulses, in this case the coefficient is a function
        #print('b')
        H_pump_L = -1j*np.sqrt(γ)*(σ_L - σ_Ld)

        H_pump_R = -1j*np.sqrt(γ)*(σ_R - σ_Rd)

        return [H_QD, [H_pump_L, L_pump_coefficient], [H_pump_R, R_pump_coefficient]]
    

def Collapse_operators(γ_L = 1, γ_R = 1):
    c_ops = []
    
    if γ_L > 0:
        c_ops.append(np.sqrt(γ_L)*σ_L)
        
    if γ_R > 0:
        c_ops.append(np.sqrt(γ_R)*σ_R)
        
    return c_ops




def Hp_coeff_t(t, args):
    
    E = args['Energy']
    Γ = args['Gamma']
    pulse_profile = args['Pulse_profile']
    
    if pulse_profile == 'rising_pulse':  
        if t < 0:
            return E*(2**-0.5)*Γ*np.exp((t/2)*Γ) #Perhaps it'd be better to put (2**-0.5) explicitly in the dynamics solver
        else:
            return 0
        
    if pulse_profile == 'square_pulse':
        if t < 0:
            return  (2**-0.5)*np.sqrt(Γ/10)
        else:
            return 0
        
    if pulse_profile == 'decreasing_pulse':  
        if t < 1/Γ:
            return E*(2**-0.5)*Γ*np.exp(-(t/2)*Γ)
        else:
            return 0
        
#########################################################
#
#           Single photon pulses functions
#
#########################################################

        
# This function computes the probaibility of finding the spin in the ground state for the single photon pulse (SPP)
# It is implemented the analtical functions Maria has provided. As above, it only needs the gamma value and the field profile.
# It returns the time_range [0] and Probability [1] vectors
def Probability_excited_state_SPP(γ, pulse_profile):
    
    if pulse_profile == 'increasing':
        
        t_range = np.arange(-10/γ, 10/γ, 0.1)
        Pe = [np.exp(-γ*np.abs(t))/2 for t in t_range]
        
        return [np.array(t_range), np.array(Pe)]
        
    if pulse_profile == 'decreasing':
        
        t_range = np.arange(0, 10/γ, 0.1)
        Pe = [(γ*t)**2*np.exp(-γ*t)/2 for t in t_range]
        
        return [np.array(t_range), np.array(Pe)]

# This function computes the Spin coherence for the single photon pulse (SPP)
# It is implemented the analtical functions Maria has provided. It only needs the gamma value and the field profile.
# It returns the time_range [0] and spin coherence [1] vectors
def Spin_coherence_SPP(γ, pulse_profile): #OK
    
    if pulse_profile == 'increasing':
        t_range = np.arange(-10/γ, 10 + 0.1, 0.1)
        s_vec = []
        for t in t_range:
            if t < 0:
                s_vec.append(2*(1 - np.exp(γ*t))/(2-np.exp(γ*t)))
            else:
                aux = s_vec[-1]
                s_vec.append(aux)
        
        
        return [t_range, s_vec]
    
    if pulse_profile == 'decreasing':
        t_range = np.arange(0, 10/γ + 0.1, 0.1)
        s_vec = [(np.exp(-γ*t)*(1 + γ*t))/(1 - (γ/2)*(np.sqrt(γ)*t*np.exp(-γ*t/2))**2) for t in t_range]
        return [t_range, s_vec]
    
# This function computes the spin entropy as explained in my notes.
# The inputs are the coherence and probability vectors of the above functions.     

def Spin_vN_entropy_SPP(Coh, Pg):
    
    ent = []
    
    for i in np.arange(0, len(Coh)):
        
        if Coh[i] > 10**-13:
            if Pg[i] >= 0.9 :
                ent.append(0)
            
            else:
                ent.append(1 + ((np.abs(Coh[i]) - Pg[i])/(2*Pg[i]))*np.log2(1 - np.abs(Coh[i])/Pg[i]) - ((np.abs(Coh[i]) + Pg[i])/(2*Pg[i]))*np.log2(1 + np.abs(Coh[i])/Pg[i]))
        
        else:
            ent.append(1)
            
    return np.array(ent)