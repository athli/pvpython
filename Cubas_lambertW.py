import scipy.special
import scipy.constants
import pvlib.singlediode
import numpy

def checkCubas(i_sc, v_oc, i_mpp, v_mpp, k_i, k_v, n, a, T, G):
    ''' Derive r_s, r_sh, i_0, i_pv from manufacturer given parameters, number
        of cells, ideality, and temperature
        Use a range of ideality of [1, 1.5]
    '''
    # defining constants
    k = scipy.constants.Boltzmann
    q = scipy.constants.elementary_charge
    T_r = 298.15
    G_r = 1000
    # thermal voltage
    v_t = n*k*T/q
    # A, B, C, D are variables used in Cubas et. al
    A = a*v_t / i_mpp
    B = -1*v_mpp * (2*i_mpp-i_sc) / (v_mpp*i_sc + v_oc * (i_mpp-i_sc))
    C = -1*(2*v_mpp-v_oc) / (a*v_t) + (v_mpp*i_sc-v_oc*i_mpp) / (v_mpp*i_sc + v_oc * (i_mpp-i_sc))
    D = (v_mpp-v_oc)/(a*v_t)

    # accounting for variation in temperature when k_v is not a percentage
    # TODO: Test if dividing by 100 fixes problems with Kim Data
    '''v_oc = v_oc*(1+(T-T_r)*(k_v/v_oc)/100)
    v_mpp = v_mpp*(1+(T-T_r)*(k_v/v_mpp)/100)
    i_sc = i_sc*(1+(T-T_r)*(k_i/i_sc)/100)*(G/G_r)
    i_mpp = i_mpp*(1+(T-T_r)*(k_i/i_mpp)/100)'''

    # accounting for variation in temperature when k_v is a percentage
    v_oc = v_oc*(1+(T-T_r)*(k_v/100))
    v_mpp = v_mpp*(1+(T-T_r)*(k_v/100))
    i_sc = i_sc*(1+(T-T_r)*(k_i/100))*(G/G_r)
    i_mpp = i_mpp*(1+(T-T_r)*(k_i/100))

    # calculating parameters
    r_s = A*(scipy.special.lambertw(B*numpy.exp(C), -1)-(D+C))
    r_sh = (v_mpp-i_mpp*r_s)*(v_mpp-r_s*(i_sc-i_mpp)-(a*v_t)) / ((v_mpp-i_mpp*r_s)*(i_sc-i_mpp) - (a*v_t*i_mpp))
    i_0 = ((r_sh+r_s)*i_sc-v_oc)/(r_sh*numpy.exp(v_oc/(a*v_t)))
    i_pv = ((r_sh+r_s) / r_sh * i_sc)

    # create dictionary of parameters for ease of use
    result = {}
    result["r_s"] = r_s
    result["r_sh"] = r_sh
    result["i_0"] = i_0
    result["i_pv"] = i_pv

    return result

def checkPVLib():
    ''' Derive i_sc, v_oc, i_mpp, v_mpp, p_mpp from solar cell parameters
    '''
