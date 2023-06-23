#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:49:55 2023

@author: brunogoes
"""

#%% PREAMBLE

# Qutip
from qutip import *
from qutip.piqs import *

# Scipy
import scipy as sp
import scipy.integrate as integrate
import scipy.special as special
import scipy.sparse as spr
from scipy.sparse.linalg import eigs
from scipy.integrate import quad, dblquad

# Matplotlib
import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Math
import numpy as np
import math as mt
from scipy.sparse.linalg import eigs
import csv
import sympy as sym
from sympy import Matrix
from sympy.solvers import solve
from sympy import Symbol #symbolic math
from sympy import *

# To do an animation of the Bloch sphere
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
import pickle
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# To index plots smartly and automatically
import string
alphabet = list(string.ascii_lowercase)

############################################
#
#   Short hand necessary operators 
#
############################################

Sx = Qobj([[0. , 1., 0., 0.], [1. , 0., 0., 0.], [0. , 0., 0., 1j], [0. , 0., -1j, 0.]])
Sy = Qobj([[0. , 0., 1., 0.], [0. , 0., 0., -1j], [1. , 0., 0., 0.], [0. , 1j, 0., 0.]])
Sz = Qobj([[0. , 0., 0., 1.], [0. , 0., 1j, 0.], [0. , -1j, 0., 0.], [1. , 0., 0., 0.]])

Jm = Qobj([[0.5 , 0., 0., -0.5], [0. , 0., 0., 0.], [0. , 0., 0., 0.], [0.5 , 0., 0., -0.5]])

sigma_moins = (Sx - 1j*Sy)/2
sigma_plus = (Sx + 1j*Sy)/2

#%% FUNCTIONS

#%% Main functions - don't modify

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
    
    
    return [real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:]]

############################################
#   Function definitions - ok
############################################

############################################
# Renormalized Rabi frequency
############################################
def Omega(alpha, g): # ok
    '''
    

    Parameters
    ----------
    alpha : Real value.
        This is the field amplitude.
    g : Real values.
        Interaction strength.

    Returns
    -------
    Real values
        The renormalized Rabi frequency.

    '''
    return 2*(g*alpha**2 -(g/4)**2)**0.5


############################################
#%%% Coefficients
############################################

# No photon re-emission coefficient ground state
def f0g(alpha, g, tau): # ok
    
    exp_factor = np.exp(-g*tau / 4)
    
    Om = Omega(alpha, g)
    
    arg = Om*tau/2
    
    soma = np.cos(arg) + (g/(2*Om))*np.sin(arg)
    
    result = exp_factor*soma
    
    return result

# No photon re-emission coefficient excited state
def f0e(alpha, g, tau): # ok
    
    exp_factor = np.exp(-g*tau / 4)
    
    Om = Omega(alpha, g)
    
    arg = Om*tau/2
    
    soma = (2*g**0.5*alpha/Om)*np.sin(arg)
    
    result = exp_factor*soma
    
    return result

# One photon re-emission coefficient ground state
def f1g(alpha, g, tau, t): # ok
    
    result = (g**0.5)*f0e(alpha, g, t)*f0g(alpha, g, tau - t)
    
    return result

# One photon re-emission coefficient excited state
def f1e(alpha, g, tau, t): # ok
    
    result = (g**0.5)*f0e(alpha, g, t)*f0e(alpha, g, tau - t)
    
    return result

# Two photons re-emission coefficient ground state
def f2g(alpha, g, tau, t,y): # ok
    
    result = g*f0e(alpha, g, t)*f0e(alpha, g, y - t)*f0g(alpha, g, tau - y)
    
    return result

# Two photons re-emission coefficient excited state
def f2e(alpha, g, tau, t, y): # ok
    
    
    result =  g*f0e(alpha, g, t)*f0e(alpha, g, y - t)*f0e(alpha, g, tau - y)
    
    return result

############################################
#%%% Evolution matrices - ok
############################################

def U_forward(gamma, delta, Omega, Gamma, t, T): #ok
    
    matrix = Qobj([[0. , 0., 0., 0.], [0. , -(gamma + 2*Gamma), -2.*delta, -2.*Omega], [0. , 2.*delta, -(gamma + 2*Gamma), 0.], [-2.*gamma , 2.*Omega, 0., -2.*gamma]])
    
    evol = (matrix*(t / 2)).expm()
    
    return evol


def U_backward(gamma, delta, Omega, Gamma, t, T): #ok
    
    matrix = Qobj([[0. , 0., 0., -2.*gamma], [0. , -(gamma + 2*Gamma), 2.*delta, 2.*Omega], [0. , -2.*delta, -(gamma + 2*Gamma), 0.], [0., -2.*Omega, 0., -2.*gamma]])
    
    evol = (matrix*(T - t) / 2).expm()
    
    return evol


############################################
#%%% Traces - ok
############################################

def tracemoins(gamma, delta, Omega, Gamma, t, T, n0, mT):
  
    Uf = U_forward(gamma, delta, Omega, Gamma, t, T)
    
    Ub = U_backward(gamma, delta, Omega, Gamma, t, T)
 
    numerator = ((Uf*n0).trans()*sigma_moins*(Ub*mT)).tr()
    
    denominator = ((Uf*n0).trans()*(Ub*mT)).tr()
    
    result = numerator/denominator
    
    return result

def traceplus(gamma, delta, Omega, Gamma, t, T, n0, mT):
    
    Uf = U_forward(gamma, delta, Omega, Gamma, t, T)
    
    Ub = U_backward(gamma, delta, Omega, Gamma, t, T)
    
    numerator = ((Uf*n0).trans()*sigma_plus*(Ub*mT)).tr()

    denominator = ((Uf*n0).trans()*(Ub*mT)).tr()
    
    result = numerator/denominator
    
    return result

def traceoffsetted(gamma, delta, Omega, Gamma, t, T, n0, mT):

    
    return 2*tracemoins(gamma, delta, Omega, Gamma, t, T, n0, mT) - conj(traceplus(gamma, delta, Omega, Gamma, t, T, n0, mT))

def tracex(gamma, delta, Omega, Gamma, t, T, n0, mT):
    
    Uf = U_forward(gamma, delta, Omega, Gamma, t, T)
    
    Ub = U_backward(gamma, delta, Omega, Gamma, t, T)
    
    numerator = ((Uf*n0).trans()*Sx*(Ub*mT)).tr()

    denominator = ((Uf*n0).trans()*(Ub*mT)).tr()
    
    result = numerator/denominator
    
    return result

def tracey(gamma, delta, Omega, Gamma, t, T, n0, mT):
    
    Uf = U_forward(gamma, delta, Omega, Gamma, t, T)
    
    Ub = U_backward(gamma, delta, Omega, Gamma, t, T)
    
    numerator = ((Uf*n0).trans()*Sy*(Ub*mT)).tr()

    denominator = ((Uf*n0).trans()*(Ub*mT)).tr()
    
    result = numerator/denominator
    
    return result

def tracez(gamma, delta, Omega, Gamma, t, T, n0, mT):

    
    Uf = U_forward(gamma, delta, Omega, Gamma, t, T)
    
    Ub = U_backward(gamma, delta, Omega, Gamma, t, T)
    
    numerator = ((Uf*n0).trans()*Sz*(Ub*mT)).tr()

    denominator = ((Uf*n0).trans()*(Ub*mT)).tr()
    
    result = numerator/denominator
    
    return result

def traceJm(gamma, delta, Omega, Gamma, t, T, n0, mT):
    
    Uf = U_forward(gamma, delta, Omega, Gamma, t, T)
    
    Ub = U_backward(gamma, delta, Omega, Gamma, t, T)
    
    numerator = ((Uf*n0).trans()*Jm*(Ub*mT)).tr()

    denominator = ((Uf*n0).trans()*(Ub*mT)).tr()
    
    result = numerator/denominator
    
    return result


#%%% Necessary tilde-coefficients 


def WignerTildeCoeffs(theta, tau, gamma, Gamma, rho0 = ket2dm(ket('1')), convergence_parameter = 0):
    '''
    

    Parameters
    ----------
    theta : Real value between 0 and pi rads.
    tau : Measurement time
    gamma : decay rate
    Gamma : 
    rho0 :Initial density matrix. The default is ket2dm(ket('1')).
    convergence_parameter : The default is 0.

    Returns
    -------
    list
        Pg, Pe, f1gtilde, f1etilde, f2gtilde, f2etilde are the necessary parameters to compute the Wigner function and density matrix.

    '''
    # Here we're using the densities of modes being equal to the time interval of the pulse.
    dg = de = tau
    
    g = gamma
    omega = theta / tau
    
    alpha = omega/(2*g**0.5)
    
    H = - (omega/2)*sigmay()
    
    tlist = np.arange(0, tau + 0.2, 0.2)
    dyn = mesolve(H, rho0, tlist, [gamma**0.5*sigmam()])
    
    Pe = (dyn.states[-1]*sigmap()*sigmam()).tr() #ok
    Pg = (dyn.states[-1]*sigmam()*sigmap()).tr() #ok
    
    func1g = lambda t : f1g(alpha, g, tau, t)
    f1gtilde = dg**(-0.5)*quad(func1g, 0, tau)[0]
    
    func1e = lambda t : f1e(alpha, g, tau, t)
    f1etilde = de**(-0.5)*quad(func1e, 0, tau)[0]
    
    func2g = lambda t, y : f2g(alpha, g, tau, t,y)
    f2gtilde = dg**(-1)*dblquad(func2g, 0, tau, lambda t: 0, lambda t: t)[0]
    
    func2e = lambda t, y : f2e(alpha, g, tau, t, y)
    f2etilde = de**(-1)*dblquad(func2e, 0, tau, lambda t: 0, lambda t: t)[0]
    
    
    return [Pg, Pe, f1gtilde, f1etilde, f2gtilde, f2etilde] 

#%%% Analytical density matrices


############################################
# Analytical density matrices
############################################

def Analytical_density_matrices(theta, tau, gamma, Gamma, rho0 = ket2dm(ket('1')), N_fock = 5, convergence_parameter = 0): 
    '''
    

    Parameters
    ----------
   theta : Real value between 0 and pi rads.
   tau : Measurement time
   gamma : decay rate
   Gamma :
   rho0 : Density matrix, optional
       The default is ket2dm(ket('1')).
    N_fock : Integer, the truncation of the Fock space, optional
         The default is 5.
    convergence_parameter :  The default is 0.

    Returns
    -------
    list
        IF THE TRACE OF THE DENSITY MATRIX IS >0.99 IT RETURNS ONLY THE DENSITY MATRICES rho_g,rho_e.
        IF TR<0.99, IT RETURNS THE DENSITY MATRICES + ERROR 1-TR

    '''
    
    # This function is precisely matching D3 of the paper, but there there is a f0epsilon tilde that is not defined at all, I believe it's a typo, if not it might be a source of error.
    
    
    dg = de = tau
    # This function is precisely D3
    g = gamma
    omega = theta / tau
    #print(omega)
    alpha = omega/(2*g**0.5)
    
    [Pg, Pe, f1gtilde, f1etilde, f2gtilde, f2etilde] = WignerTildeCoeffs(theta, tau, gamma, Gamma, rho0, convergence_parameter)
    
    coeff00g = f0g(alpha, g, tau)*np.conj(f0g(alpha, g, tau)) #Pg + 2*f2gtilde*np.conj(f2gtilde) + f1gtilde*np.conj(f1gtilde)
    
    coeff11g = f1gtilde*np.conj(f1gtilde)
    
    coeff22g = 2*f2gtilde*np.conj(f2gtilde)
    
    coeff12g = -2**0.5*np.conj(f2gtilde)*f1gtilde
       
    coeff02g = 2**0.5*np.conj(f2gtilde)*f0g(alpha, g, tau)
    
    coeff01g = -np.conj(f1gtilde)*f0g(alpha, g, tau)
    
    ##################
    
    coeff00e = f0e(alpha, g, tau)*np.conj(f0e(alpha, g, tau))#Pe + 2*f2etilde*np.conj(f2etilde) + f1etilde*np.conj(f1etilde)
    
    coeff11e = f1etilde*np.conj(f1etilde)
    
    coeff22e = 2*f2etilde*np.conj(f2etilde)
    
    coeff12e = -2**0.5*np.conj(f2etilde)*f1etilde
       
    coeff02e = 2**0.5*np.conj(f2etilde)*f0e(alpha, g, tau)
    
    coeff01e = -np.conj(f1etilde)*f0e(alpha, g, tau)
    
    el00 = basis(N_fock, 0)*basis(N_fock, 0).dag()
    el11 = basis(N_fock, 1)*basis(N_fock, 1).dag()
    el22 = basis(N_fock, 2)*basis(N_fock, 2).dag()
    el12 = basis(N_fock, 1)*basis(N_fock, 2).dag()
    el02 = basis(N_fock, 0)*basis(N_fock, 2).dag()
    el01 = basis(N_fock, 0)*basis(N_fock, 1).dag()  
                        
    rho_g = (coeff00g*el00 + coeff11g*el11 + coeff22g*el22 + (coeff12g*el12 + (coeff12g*el12).dag()) + (coeff02g*el02 + (coeff02g*el02).dag()) + (coeff01g*el01 + (coeff01g*el01).dag()))/Pg
    
    rho_e = (coeff00e*el00 + coeff11e*el11 + coeff22e*el22 + (coeff12e*el12 + (coeff12e*el12).dag()) + (coeff02e*el02 + (coeff02e*el02).dag()) + (coeff01e*el01 + (coeff01e*el01).dag()))/Pe
    
    trg = rho_g.tr()
    tre = rho_e.tr()
    
    if trg >= 0.99 and tre >=0.99:
        
        return [rho_g, rho_e, 0,0]
    else:
        
        print('WARNING: There is a problem with the normalization, the trace of rho_g is='+str(trg)+'and rho_e is='+str(tre))
        err_trg = 1 - trg
        err_tre = 1 - tre
        
        return [rho_g, rho_e, err_trg, err_tre]
    
#%%% Number of excitations


def NumberOfExcitations_CentralFrequency(theta, tau, gamma, Gamma, rho0 = ket2dm(ket('1')), convergence_parameter = 0):
    # This function is precisely D3 of the Appendix of the paper.
    g = gamma
    omega = theta / tau
  
    alpha = omega/(2*gamma**0.5)
    
    alpha0 = (alpha*np.conj(alpha)*tau)**0.5
    
    [Pg, Pe, f1gtilde, f1etilde, f2gtilde, f2etilde] = WignerTildeCoeffs(theta, tau, gamma, Gamma, rho0, convergence_parameter)
    
    
    line1g = (Pg - f1gtilde**2 - 2*f2gtilde**2)*(alpha0**2)
    line2g = f1gtilde**2*(1 + alpha0**2)
    line3g = + 2*f2gtilde**2*(2 + alpha0**2)
    line4g = -8*np.real(np.conj(f2gtilde)*f1gtilde)*(alpha0/2)
    line6g = -4*(alpha0/2)*np.real(f0g(alpha, g, tau)*f1gtilde)
    
    
    line1e = (Pe - f1etilde**2 - 2*f2etilde**2)*(alpha0**2)
    line2e = f1etilde**2*(1 + alpha0**2)
    line3e = + 2*f2etilde**2*(2 + alpha0**2)
    line4e = -8*np.real(np.conj(f2etilde)*f1etilde)*(alpha0/2)
    line6e = -4*(alpha0/2)*np.real(f0e(alpha,g,tau)*f1etilde)
    
    Ng = (line1g + line2g + line3g + line4g + line6g)/Pg
    Ne = (line1e + line2e + line3e + line4e + line6e)/Pe
    
    return [Ng - alpha0**2, Ne - alpha0**2]

def NumberOfExcitations_Total(theta, tau, gamma, Gamma, rho0 = ket2dm(ket('1')), Two_photons = False):
    
    omega = theta / tau

    alpha = omega/(2*gamma**0.5)

    H = - (omega/2)*sigmay()

    tlist = np.arange(0, tau + 0.2, 0.2)
    dyn = mesolve(H, rho0, tlist, [gamma**0.5*sigmam()])

    Pe = (dyn.states[-1]*sigmap()*sigmam()).tr() #ok
    Pg = (dyn.states[-1]*sigmam()*sigmap()).tr() #ok

    func1e = lambda t: f1e(alpha, gamma, tau, t)**2
    p1e = quad(func1e, 0, tau)[0] #ok
    
 
    func10e = lambda t: f0e(alpha, gamma, tau)*f1e(alpha, gamma, tau, t)
    p01e = quad(func10e, 0, tau)[0] #ok   
 
    func1g = lambda t: f1g(alpha, gamma, tau, t)**2
    p1g = quad(func1g, 0, tau)[0] #ok
    
    func10g = lambda t: f0g(alpha, gamma, tau)*f1g(alpha, gamma, tau, t)
    p01g = quad(func10g, 0, tau)[0] #ok
    
    if Two_photons == False:
        
        ne = (Pe*alpha**2*tau + p1e - 2*alpha*p01e)/Pe
        ng = (Pg*alpha**2*tau + p1g - 2*alpha*p01g)/Pg
        
        return np.real([ng - alpha**2*tau, ne - alpha**2*tau])
    
    else:
        
        p2e = dblquad(lambda t, y : f2e(alpha, gamma, tau, t, y)**2, 0, tau, lambda t : 0, lambda t : t)[0]
        p2g = dblquad(lambda t, y : f2g(alpha, gamma, tau, t, y)**2, 0, tau, lambda t : 0, lambda t : t)[0]
        
        
        aux1 = dblquad(lambda t, y : f2e(alpha, gamma, tau, t, y)*f1e(alpha, gamma, tau, t), 0, tau, lambda t : 0, lambda t : t)[0]
        aux2 = dblquad(lambda t, y : f2e(alpha, gamma, tau, t, y)*f1e(alpha, gamma, tau, y), 0, tau, lambda t : 0, lambda t : t)[0]
        p12e = aux1 + aux2
        
        aux3 = dblquad(lambda t, y : f2g(alpha, gamma, tau, t, y)*f1g(alpha, gamma, tau, t), 0, tau, lambda t : 0, lambda t : t)[0]
        aux4 = dblquad(lambda t, y : f2g(alpha, gamma, tau, t, y)*f1g(alpha, gamma, tau, y), 0, tau, lambda t : 0, lambda t : t)[0]
        p12g = aux3 + aux4
        
        ne = (Pe*alpha**2*tau + p1e + p2e - 2*alpha*(p01e + p12e))/Pe
        ng = (Pg*alpha**2*tau + p1g + p2g - 2*alpha*(p01g + p12g))/Pg
               
        return np.real([ng - alpha**2*tau, ne - alpha**2*tau])

#%%% Wigner functions

###########################################
#   Number values - 1 and 2 photons
###########################################

# This function is precisely D4 of the paper, with the densities fixed to tau:
def Wf_g_and_e2_FixedDensities(theta, u, v, tau, gamma, Gamma, rho0 = ket2dm(ket('1')), convergence_parameter = 0):
    
    de = dg = tau # the densities are precisely the duration of the pulse.
    g = gamma
    omega = theta / tau
    
    alpha = omega/(2*gamma**0.5)
    
    [Pg, Pe, f1gtilde, f1etilde, f2gtilde, f2etilde] = WignerTildeCoeffs(theta, tau, gamma, Gamma, rho0)   

    prefactor = 2*np.exp(-2*(u**2 + v**2))/np.pi
    
    line1g = Pg - f1gtilde**2 - 2*f2gtilde**2
    line2g = -f1gtilde**2*(1- 4*(u**2 +v**2)) + 2*f2gtilde**2*(1 - 8*v**2 + 8*((-1 + u)*u + v**2)*(u + u**2 + v**2))
    line3g = -8*np.real(np.conj(f2gtilde)*f1gtilde*(u-1j*v))*(2*(u**2 + v**2)-1)
    line4g = 8*np.real(np.conj(f2gtilde)*f0g(alpha, g, tau)*(u+1j*v)**2)
    line5g = -4*np.real(f0g(alpha, g, tau)*np.conj(f1gtilde)*(u+1j*v))
    
    line1e = Pe - f1etilde**2 - 2*f2etilde**2
    line2e = -f1etilde**2*(1- 4*(u**2 +v**2)) + 2*f2etilde**2*(1 - 8*v**2 + 8*((-1 + u)*u + v**2)*(u + u**2 + v**2))
    line3e = -8*np.real(np.conj(f2etilde)*f1etilde*(u-1j*v))*(2*(u**2 + v**2)-1)
    line4e = 8*np.real(np.conj(f2etilde)*f0e(alpha, g, tau)*(u+1j*v)**2)
    line5e = -4*np.real(f0e(alpha, g, tau)*np.conj(f1etilde)*(u+1j*v))
                         
    Wg = prefactor*(line1g + line2g + line3g + line4g + line5g)/Pg
    
    We = prefactor*(line1e + line2e + line3e + line4e + line5e)/Pe
    
    
    return [Wg, We]

def Wf_g_and_e2Exact(theta, u, v, tau, gamma, Gamma, rho0 = ket2dm(ket('1')), convergence_parameter = 0):
    de = dg = tau # the densities are precisely the duration of the pulse.
    g = gamma
    omega = theta / tau
    
    alpha = omega/(2*gamma**0.5)
    
    [Pg, Pe, f1gtilde, f1etilde, f2gtilde, f2etilde] = WignerTildeCoeffs(theta, tau, gamma, Gamma, rho0)
     

    prefactor = 2*np.exp(-2*(u**2 + v**2))/np.pi
    
    line1g = Pg - f1gtilde**2 - 2*f2gtilde**2
    line2g = -f1gtilde**2*(1- 4*(u**2 +v**2)) + 2*f2gtilde**2*(1 - 8*v**2 + 8*((-1 + u)*u + v**2)*(u + u**2 + v**2))
    line3g = -8*np.real(np.conj(f2gtilde)*f1gtilde*(u-1j*v))*(2*(u**2 + v**2)-1)
    line4g = 8*np.real(np.conj(f2gtilde)*f0g(alpha, g, tau)*(u+1j*v)**2)
    line5g = -4*np.real(f0g(alpha, g, tau)*np.conj(f1gtilde)*(u+1j*v))
    
    line1e = Pe - f1etilde**2 - 2*f2etilde**2
    line2e = -f1etilde**2*(1- 4*(u**2 +v**2)) + 2*f2etilde**2*(1 - 8*v**2 + 8*((-1 + u)*u + v**2)*(u + u**2 + v**2))
    line3e = -8*np.real(np.conj(f2etilde)*f1etilde*(u-1j*v))*(2*(u**2 + v**2)-1)
    line4e = 8*np.real(np.conj(f2etilde)*f0e(alpha, g, tau)*(u+1j*v)**2)
    line5e = -4*np.real(f0e(alpha, g, tau)*np.conj(f1etilde)*(u+1j*v))
                         
    Wg = prefactor*(line1g + line2g + line3g + line4g + line5g)/Pg
    
    We = prefactor*(line1e + line2e + line3e + line4e + line5e)/Pe
    
    dxdy = (u[1]-u[0])*(v[1]-v[0])
    
    normg = sum(Wg*dxdy)
    norme = sum(We*dxdy)
    
    if normg >=0.9 and norme >=0.9:
        return [Wg, We]
    else:
        print('WARNING: There is a problem with the normalization, the integral of Wg is='+str(normg)+'and We is='+str(norme))
        [err_normg, err_norme] = [1-normg, 1-norme]
        
        return [Wg, We, err_normg, err_norme]

print('hello world!')
