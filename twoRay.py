#!/usr/bin/env python3

from math import sqrt, atan2, pi, sin, cos, log10
from scipy.constants import c
import cmath
import numpy as np
import matplotlib.pyplot as plt

'''
First define some methods that help compute the geometry of the two-ray model based on:
1. Height of transmitter (ht)
2. Height of receiver (hr)
3. Ground distance between Tx and Rx (d)
'''


def d_0(d, ht, hr):
	'''
	Calculates the line of sight distance from t to r
	'''
	return sqrt((ht - hr)**2 + d**2)


def d_1(d, ht, hr):
	'''
	Calculates the total distance travelled by the reflected ray
	'''
	return sqrt((ht + hr)**2 + d**2)


def theta(d, ht, hr):
  '''
  Calculates the angle of reflection (relative to the ground)
  '''
  return atan2(ht*(1+ht/hr),d)


def dPhase(d, ht, hr, f):
	'''
	Calculates phase difference between the main ray and reflected ray given the geometry and
	the frequency of the signal.
	'''
	lam = c/f
	x = (2*pi/lam*(d_1(d, ht, hr) - d_0(d, ht, hr)))%(2*pi)
	if x > pi:
		return x-2*pi
	return x


def dSpread(d, ht, hr):
	'''
	Calculates delay spread (difference in arrival time) between the main ray (los) and the
  reflected ray - assuming the speed of light in a vaccuum.
	'''
	return (d_1(d, ht, hr) - d_0(d, ht, hr))/c

# Assume fixed permittivity of earth, however a frequency dependent one could be used, as well
# as models for other types of surfaces like sea water etc....
perm_earth = 15
def R(d, ht, hr):
  '''
  Calculates the reflection coefficient due to the angle of arrival of the reflectd
  beam to the surface it is reflecting from.
  '''
  theta_ = theta(d, ht, hr)
  Z = sqrt(perm_earth - cos(theta_)**2)
  return (sin(theta_) - Z)/(sin(theta_)+Z)


def two_ray(d_m, h_t, h_r, f, f_c=-1):
	'''
	Put all of it together to get the two ray path loss.
	'''
	d0 = d_0(d_m, h_t, h_r)
	d1 = d_1(d_m, h_t, h_r)
	R_ = R(d_m, h_t, h_r)
	dPhi = dPhase(d_m, h_t, h_r, f)
	x = cmath.exp(complex(0, -dPhi))
	lam = c/f_c if f_c > 0 else c/f
	response = lam/(4*pi) * (1/d0 + R_*x/d1)
	#loss = (lam/(4*pi))**2 * abs(1/d0 + R_*x/d1)**2
	return response


def two_ray_pl(d_m, h_t, h_r, f, f_c=-1, returnDb=True):
	response = two_ray(d_m, h_t, h_r, f, f_c)
	loss = response * response.conjugate()
	if returnDb:
		return -10*log10(loss)
	else:
		return loss

def wideband_two_ray(f_c, bw, d_m, h_t, h_r, returnDb=True, numSubBands=2048):
	'''
	Computes an aggregate frequency response taking into account a signal that is spead spectrum and
	is able to account for frequency selective gains and losses across a wide bandwidth.
	'''
	# Assume some 'subbands', each of which have their own independent two-ray reflected beam
	freqs = np.arange(f_c-bw/2, f_c+bw/2, bw/numSubBands)
	freq_response = np.array([two_ray(d_m, h_t, h_r, f, f_c=f) for f in freqs])
	return freqs, freq_response

def wideband_two_ray_pl(f_c, bw, d_m, h_t, h_r, returnDb=True, numSubBands=2048, plotLabel=''):
	'''
	Computes an aggregate path loss taking into account a signal that is spead spectrum and
	is able to account for frequency selective gains and losses across a wide bandwidth.
	'''
	# Assume some 'subbands', each of which have their own independent two-ray reflected beam
	freqs = np.arange(f_c-bw/2, f_c+bw/2, bw/numSubBands)
	pls = np.array([two_ray_pl(d_m, h_t, h_r, f, f_c=f, returnDb=returnDb) for f in freqs])
	if plotLabel != '':
		plt.plot(freqs, pls, label=plotLabel)
	return pls

def wideband_pl(f_c, bw, d_m, h_t, h_r, returnDb=True):
	p = wideband_two_ray_pl(f_c, bw, d_m, h_t, h_r, returnDb=returnDb)
	return np.average(p)

if __name__ == "__main__":

	bw = 56e6
	ht1 = 9
	ht2 = 13.7
	hr = 3810
	f = 1815e6

	# Used to debug above routines
	dist = np.arange(1000, 60000, 100)

	plt.plot(dist, [dSpread(dd, ht2, hr) for dd in dist])
	plt.title(f'Delay Spread vs distance @ {f/1e6} MHz')
	plt.ylabel('Delay (s)')
	plt.xlabel('Distance from base (m)')
	#plt.ylim(0, 5e-8)
	#plt.xlim(0, 2500)
	plt.show()

	ph_t_r = [dPhase(dd, ht2, hr, f) for dd in dist]
	plt.plot(dist, ph_t_r, label='Tx-Rx')
	plt.title(f'Delta phase vs distance @ {f/1e6} MHz')
	plt.ylabel('Phase')
	plt.xlabel('Distance from base (m)')
	plt.legend()
	plt.show()

	for dis in [40000, 41000, 42000]:
		freqs, response = wideband_two_ray(f, bw, dis, ht1, hr)
		plt.plot(freqs, np.abs(response), label=f'ht = {ht1}, hr = {hr}, freq response @ d = {dis} m')
	plt.legend()
	plt.show()

	loss_v_d_1 = np.array([wideband_pl(f, bw, d, ht1, hr) for d in dist])
	loss_v_d_2 = np.array([wideband_pl(f, bw, d, ht2, hr) for d in dist])
	plt.plot(dist, loss_v_d_1, label=f'two-ray pathloss (fc = {f}, ht = {ht1}m, hr={hr}m)')
	plt.plot(dist, loss_v_d_2, label=f'two-ray pathloss (fc = {f}, ht = {ht2}m, hr={hr}m)')
	plt.ylabel('Path loss (dB)')
	plt.xlabel('Distance from base (m)')
	plt.legend()
	plt.show()

	x = wideband_two_ray_pl(f, 56e6, 10e3, 15e3, 15e3, plotLabel='d=10000, alt=15k')
	x = wideband_two_ray_pl(f, 56e6, 10100, 15e3, 15e3, plotLabel='d=10100, alt=15k')
	x = wideband_two_ray_pl(f, 56e6, 10500, 15e3, 15e3, plotLabel='d=10500, alt=15k')
	plt.legend()
	plt.show()
