import os, sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import cholesky

try:
    import lsqfit
    import gvar as gv
except:
    sys.exit('you must install lsqfit for this code to work\npip install lsqfit')
import utils

import numpy as np
import gvar as gv

p0 = {'p_0':1.5}
for n in range(1,20):
    p0['p_%d' %n] = 0.1

def make_p0(n,seed=None):
    np.random.seed(seed)
    p0 = {'p_0':1.5}
    for i in range(1,n+1):
        p0['p_%d' %i] = np.random.normal(0,0.5)
    return p0

priors = {'p_0':gv.gvar(1.5, 1)}
for n in range(1,20):
    priors['p_%d' %n] = gv.gvar(0,1)

def polynomial(x,p):
    ''' x = independent parameters
        p = dictionary of coefficients, p_0, p_1, ...
        y = p_0 + p_1 * x + p_2 * x**2 + ...
    '''
    y = 0
    for k in p:
        n = int(k.split('_')[1])
        y += p[k] * x**n
    return y

def add_noise(x, mu=0., sig=1., Nsamp=100, seed=None):
    ''' add random noise for all points x
        mu    : mean of noise
        sig   : width of noise
        Nsamp : number of random samples
        seed  : seed the random number generator with an int
    '''
    np.random.seed(seed)
    noise = np.random.normal(loc=mu, scale=sig, size=(Nsamp, x.shape[0]))

    return noise

def main():
    parser = argparse.ArgumentParser(
        description='Perform sample linear regression')
    ''' data options '''
    parser.add_argument('--data_n',        type=int, default=1,
                        help=              'order of polynomial to generate data [%(default)s]')
    parser.add_argument('--seed',          type=int, default=None,
                        help=              'integer seed for numpy random number generator [%(default)s]')
    parser.add_argument('--mu',            type=float, default=0,
                        help=              'mean of random noise [%(default)s]')
    parser.add_argument('--sig',           type=float, default=1,
                        help=              'width of random noise [%(default)s]')
    parser.add_argument('--Nsamp',         type=int,   default=10000,
                        help=              'Number of random samples for noise [%(default)s]')

    ''' fitting options '''
    parser.add_argument('--fit_n',         type=int, default=1,
                        help=              'order of polynomial to fit data [%(default)s]')
    parser.add_argument('--freq_fit',      default=True, action='store_false',
                        help=              'Perform numerical frequentist fit? [%(default)s]')
    parser.add_argument('--bayes_fit',     default=False, action='store_true',
                        help=              'Perform Bayes fit? [%(default)s]')
    parser.add_argument('--linear',        default=False, action='store_true',
                        help=              'use VarPro to do linear regression? [%(default)s]')
    parser.add_argument('--add_corr',      type=float, default=None,
                        help=              'create correlation in data? [%(default)s]')
    parser.add_argument('--uncorr',        default=False, action='store_true',
                        help=              'uncorrelate data? [%(default)s]')
    parser.add_argument('--reveal',        default=False, action='store_true',
                        help=              'reveal params used to make data? [%(default)s]')

    ''' plotting options '''
    parser.add_argument('--show_plots',    default=True, action='store_false',
                        help=              'Show plots?  [%(default)s]')
    parser.add_argument('--interact',      default=False, action='store_true',
                        help=              'open IPython instance after to interact with results? [%(default)s]')


    args = parser.parse_args()
    print(args)

    p0 = make_p0(args.data_n, seed=args.seed)

    ''' select parameters to generate data '''
    p_data = {k:v for k,v in p0.items() if int(k.split('_')[1]) <= args.data_n}

    ''' generate noisy data '''
    x = np.arange(0,1.02,.02)
    y = polynomial(x,p_data)
    g_noise = add_noise(x, mu=args.mu, sig=args.sig, Nsamp=args.Nsamp, seed=args.seed)
    y_g = np.zeros_like(g_noise)
    for i,d in enumerate(y):
        y_g[:,i] = d + g_noise[:,i]
    y_gv = gv.dataset.avg_data(y_g)

    ''' create correlation in the data? '''
    if args.add_corr is not None:
        n_d = y_gv.shape[0]
        cov  = gv.evalcov(y_gv)
        cov2 = np.zeros_like(cov)
        for i in range(n_d):
            for j in range(i+1):
                if i == j:
                    cov2[i,j] = cov[i,j]
                else:
                    cov2[i,j] = cov[i,j] + args.add_corr * cov[i,i]
                    cov2[j,i] = cov[i,j] + args.add_corr * cov[i,i]
        y_gv = gv.gvar([k.mean for k in y_gv], cov2)

    ''' do numerical fits '''
    p_fit = {k:v for k,v in p0.items() if int(k.split('_')[1]) <= args.fit_n}
    linear=[]
    if args.linear:
        linear = [k for k in p_fit]
    if args.freq_fit:
        print('---------------------------------------------------------------')
        print('       Frequentist Fit')
        print('---------------------------------------------------------------')
        if args.uncorr:
            freq_fit = lsqfit.nonlinear_fit(udata=(x, y_gv), p0=p_fit, fcn=polynomial, linear=linear)
        else:
            freq_fit = lsqfit.nonlinear_fit( data=(x, y_gv), p0=p_fit, fcn=polynomial, linear=linear)
        print(freq_fit)
    if args.bayes_fit:
        print('---------------------------------------------------------------')
        print('       Bayes Fit')
        print('---------------------------------------------------------------')
        prior = {k:v for k,v in priors.items() if int(k.split('_')[1]) <= args.fit_n}
        if args.uncorr:
            bayes_fit = lsqfit.nonlinear_fit(udata=(x, y_gv), prior=prior, fcn=polynomial, linear=linear)
        else:
            bayes_fit = lsqfit.nonlinear_fit(data=(x, y_gv), prior=prior, fcn=polynomial, linear=linear)
        print(bayes_fit)

    if args.reveal:
        print('---------------------------------------------------------------')
        print('       Underlying parameters')
        print('---------------------------------------------------------------')
        for k in p0:
            print(k, p0[k])

    if args.show_plots:
        plt.ion()
        fig = plt.figure()
        ax  = plt.axes([0.12,0.12, 0.87, 0.87])
        ''' plot fit result '''
        x_plot = np.arange(x[0],x[-1]+.1,.1)
        if args.freq_fit or args.bayes_fit:
            if args.bayes_fit:
                post = bayes_fit.p
            elif args.freq_fit:
                post = freq_fit.p
            y_freq = polynomial(x_plot, post)
            r  = np.array([k.mean for k in y_freq])
            dr = np.array([k.sdev for k in y_freq])
            ax.fill_between(x_plot, r-dr, r+dr, color='r', alpha=.4)

        ''' plot data '''
        m   = np.array([k.mean for k in y_gv])
        dm  = np.array([k.sdev for k in y_gv])
        ax.errorbar(x,m, yerr=dm,
                    linestyle='None', marker='s', mfc='None', color='k')

    if args.interact:
        import IPython; IPython.embed()
    if args.show_plots:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
