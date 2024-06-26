#!/usr/bin/env python3

from math import sqrt, atan2, pi, sin, cos, log10
from scipy.constants import c
import cmath
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


# Distance units conversion

def ft_to_m(val):
	return val * 0.3048

def m_to_ft(val):
	return val * 3.28084

def ft_to_nm(val):
	return val * 0.000164579

def nm_to_ft(val):
	return val * 6076.12

def m_to_nm(val):
	return ft_to_nm(m_to_ft(val))

def nm_to_m(val):
	return nm_to_ft(ft_to_m(val))

def nm_to_km(val):
	return nm_to_m(val) / 1000.


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
		return -10*log10(np.real(loss))
	else:
		return loss


def wideband_two_ray(f_c, bw, d_m, h_t, h_r, returnDb=True, numSubBands=128):
	'''
	Computes an aggregate frequency response taking into account a signal that is spead spectrum and
	is able to account for frequency selective gains and losses across a wide bandwidth.
	'''
	# Assume some 'subbands', each of which have their own independent two-ray reflected beam
	freqs = np.arange(f_c-bw/2, f_c+bw/2, bw/numSubBands)
	freq_response = np.array([two_ray(d_m, h_t, h_r, f, f_c=f, returnDb=returnDb) for f in freqs])
	return freqs, freq_response


def wideband_two_ray_pl(f_c, bw, d_m, h_t, h_r, returnDb=True, numSubBands=128, plotLabel=''):
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

    parser = argparse.ArgumentParser(
                    prog='twoRay.py',
                    description='Computes path loss using the simple two-ray multipath model')

    parser.add_argument('-bw', '--bandwidth', help='bandwidth of signal (hz)', type=float, default=1e6)
    parser.add_argument('-fc', '--center-freq', help='center frequency of signal (hz)', type=float, default=1e9)
    parser.add_argument('-ht', '--transmitter-height', help='height of transmitter (ft)', type=float, default=10000)
    parser.add_argument('-hr', '--receiver-height', help='height of receiver (ft)', type=float, default=10000)
    parser.add_argument('-g', '--ground-height', help='ground height above sea level (ft)', type=float, default=0)

    parser.add_argument('-ds', '--start-distance', help='start distance (nautical miles)', type=float, default=0.01)
    parser.add_argument('-de', '--end-distance', help='end distance (nautical miles)', type=float, default=50)
    parser.add_argument('--step-distance', help='distance for each plot - default = [start, end)', type=float, default=-1)
    parser.add_argument('-N', '--num-points', help='number of plot points', type=int, default=1000)
    parser.add_argument('-t', '--threshold', help='threshold path-loss (dB) ', type=float, default=[], action="append")
    parser.add_argument('-km', help='distance in km instead of nautical miles', action="store_true")
    parser.add_argument('--plot-envelope', help='Plot envelope of signal rather than actual signal', action="store_true")

    parser.add_argument('-E', '--permittivity', help=f'Relative permittivity (default = {perm_earth})', type=float, default=perm_earth)

    args = parser.parse_args()

    bw = args.bandwidth
    f = args.center_freq
    base = args.ground_height
    ht = args.transmitter_height
    hr = args.receiver_height
    perm_earth = args.permittivity

    start_ = args.start_distance
    end_ = args.end_distance if (args.step_distance == -1) else start_+args.step_distance
        
    from scipy.signal import hilbert
    from scipy.signal import butter, lfilter

    def butter_lowpass(cutOff, fs=1., order=5):
        nyq = 0.5 * fs
        normalCutoff = cutOff / nyq
        b, a = butter(order, normalCutoff, btype='low', analog = True)
        return b, a

    def butter_lowpass_filter(data, cutOff, fs=1., order=4):
        b, a = butter_lowpass(cutOff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def envelope(signal, x_axis, nseg=50):
        res = []
        res_x = []
        skip = len(signal)//nseg
        for s in range(0, len(signal), skip):
            sig = signal[s:s+skip]
            x = x_axis[s:s+skip]
            pairs = [(sig[i], x[i]) for i in range(len(sig))]
            n_pts = max(2, len(sig)//10)
            sortVals = sorted(pairs, key = lambda val: val[0], reverse=True)[:n_pts]
            result = sorted(sortVals, key=lambda val: val[1])
            envSig = [r[0] for r in result]
            envX = [r[1] for r in result]
            res.extend(envSig)
            res_x.extend(envX)
        res = np.array(res)
        res_x = np.array(res_x)
        from scipy.interpolate import make_interp_spline, BSpline
        # 300 represents number of points to make between min and max
        xnew = np.linspace(res_x.min(), res_x.max(), 1000) 
        spl = make_interp_spline(res_x, res, k=3)  # type: BSpline
        res_smooth = spl(xnew)
        return res_smooth, xnew
        
    def doPlot(args, start, end):
        numPoints = args.num_points
        delta=(end - start)/numPoints
        nm_dist = np.arange(start, end, delta)
        km_dist = nm_to_km(nm_dist)
        dist = nm_to_m(nm_dist)
        plot_dist = km_dist if args.km else nm_dist
        dist_label = "km" if args.km else "nm"
        ht_m = ft_to_m(ht - base)
        hr_m = ft_to_m(hr - base)
        loss_v_d_2 = np.array([wideband_pl(f, bw, d, ht_m, hr_m) for d in dist])
        if args.plot_envelope:
            env, envX = envelope(loss_v_d_2, plot_dist)
            plt.plot(envX, env, label=f'two-ray pathloss envelope (fc = {f}, ht = {ht} ft MSL, hr={hr} ft MSL)')
        else:
            plt.plot(plot_dist, loss_v_d_2, label=f'two-ray pathloss (fc = {f}, ht = {ht} ft MSL, hr={hr} ft MSL)')
        if args.threshold:
            for t in args.threshold:
                plt.axhline(t, color='black')
        plt.ylabel('Path loss (dB)')
        plt.xlabel(f'Distance from base ({dist_label})')
        plt.legend()
        plt.show()

    while end_ <= args.end_distance:
        doPlot(args, start_, end_)
        start_ += args.step_distance/2 if (args.step_distance != -1) else 0
        end_ += args.step_distance/2 if (args.step_distance != -1) else 1

