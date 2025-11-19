import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
import numpy as np
import copy
import os
from numba import njit
from matplotlib.collections import LineCollection
abspath = r"C:\FMF\Magisterij\Praktikum_strojnega_ucenja\PSUF_naloga3\\"

lamb = 505          #valovna dolžina [nm]
noise = 100           #šum
gamma1 = 0.098      #konstanta

I = np.load(os.path.join(abspath + 'pregenerated_data', 'intensity{}noise{}.npy'.format(lamb, noise)))      #10^5 intenzitetnih krivulj
K_gen = np.load(os.path.join(abspath + 'pregenerated_data', 'Kvalues.npy'))                                 #10^5 pripadajočih elastičnih konst.
theta0 = np.load(os.path.join(abspath + 'pregenerated_data', 'theta0.npy'))                                 #10^5 prip. zač. profilov
Kmax = 20e-12
K_norm = K_gen/Kmax
#N = len(I)      #število parov ( I(t), K )

#times = np.linspace(0, len(I[0])-1, len(I[0]))    #indeksi časov (default: [0, 399])
T = 1.2     #končni čas [s]
nt = len(I[0])      #število časovnih točk
times = np.linspace(0, T, nt)

M = len(I)
N_train = int(0.8*M)                    #velikost train (trenirnega) seta
N_val = int((M-N_train)/2)              #velikost validacijskega seta
N_test = M - N_train - N_val            #velikost testnega seta
K_train = K_norm[:N_train]
K_val = K_norm[N_train:N_train + N_val]
K_test = K_norm[N_train + N_val:]

def theta_time_evolution(theta0, C, D=10 * 1e-6, dt=5 * 1e-6, num_timesteps=240000, nth_step_save=600):
    """
    Calculates time evolution (relaxation) of director profile (given by angle theta).

    :param theta0: Starting profile of theta (array of length N)
    :param C: Relaxation constant K/gamma
    :param D: Thickness of the layer in meters
    :param dt: Timestep in seconds
    :param num_timesteps: Number of timesteps in the simulation
    :param nth_step_save: Save theta profile at every nth step (number of saves: M = num_timesteps // nth_step_save)
    :return: Time evolution of theta (array of dimensions (M, N))
    """

    N = len(theta0)
    dz = D / (N - 1)
    cnst = C * dt / (dz ** 2)

    if cnst > 0.5:
        print(cnst)
        raise ValueError("Iteration step too large, try smaller timestep or change other parameters.")

    thetas_out = np.zeros((num_timesteps // nth_step_save, N))
    theta1 = np.copy(theta0)
    for t in range(num_timesteps):
        theta1[1:N-1] += cnst * (theta1[2:N] - 2 * theta1[1:N-1] + theta1[:N-2])
        if t % nth_step_save == 0:
            thetas_out[t // nth_step_save, :] = theta1

    return thetas_out

#--------------------histogram žrebanih K-jev (ločen na trening, validacijske in testne)---------------------------------------
'''plt.figure(figsize=(7.5,5))
#plt.hist(K_norm, bins=20, fc='lime', label='izžrebane konstante')
plt.hist(K_train, bins=20, fc='navy', ec='w')
plt.hist(2, fc='navy', label='trenirne konstante')            #just to show it in the legend - without edgecolor :)
plt.hist([K_val, K_test], bins=20, color=['yellow', 'deeppink'], rwidth=0.6, label=['validacijske', 'testne'])
plt.title('Porazdelitev naključno izžrebanih elastičnih konstant')
plt.xlim(-0.05, 1.05)
plt.xlabel('$K$')
plt.ylabel('$N$')
plt.legend(bbox_to_anchor=(1.04, 0.5), loc='center left')
plt.tight_layout()
plt.show()'''
#--------------------------------------------------------------------------------------------------

#--------------------histogram žrebanih K-jev------------------------------------------------------
'''plt.figure(figsize=(7.5,5))
plt.hist(K_norm, bins=20, fc='navy', ec='w')
plt.title('Porazdelitev naključno izžrebanih elastičnih konstant')
plt.xlim(-0.05, 1.05)
plt.xlabel('$K$')
plt.ylabel('$N$')
plt.tight_layout()
plt.show()

#še v logaritemski skali:
def plot_loghist(x, bins):
    plt.figure(figsize=(7.5,5))
    hist, bins = np.histogram(x, bins=20)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 50)
    plt.hist(x, bins=logbins, fc='navy', ec='w')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$K$')
    plt.ylabel('$N$')
    plt.title('Porazdelitev naključno izžrebanih elastičnih konstant')
    plt.tight_layout()
    plt.show()

plot_loghist(K_norm, 20)'''

#--------------------------------------------------------------------------------------------------

#https://stackoverflow.com/questions/25780022/how-to-make-python-format-floats-with-certain-amount-of-significant-digits

ind = 5            #indeks - KATERO INTENZITETNO KRIVULJO RIŠEMO !!!

"""fig, ax = plt.subplots(1)
plt.plot(theta0[ind]*180/np.pi, marker='.', markersize=3, linewidth=0.7, label='K={:.3g}'.format(K_gen[0]))
plt.title('Začetni profil $\\theta(z,t=0)$')
plt.xlabel('$z$')
plt.ylabel('$\\theta$[°]')
plt.xticks([0, 99.5, 199], ['$0$', '$d/2$', '$d$'])
plt.xlim(-1, 200)
'''yticks = np.array([-np.pi/4, 0, np.pi/4])
while True:
    if yticks[0] > np.min(theta0[ind]):
        yticks = np.insert(yticks, 0, yticks[0]-np.pi/4)
    else: break
while True:
    if yticks[-1] < np.max(theta0[ind]):
        yticks = np.insert(yticks, -1, yticks[-1]+np.pi/4)
    else: break
print(yticks)
ylabels = ['${:.3g}\pi$'.format(yticks[i]/np.pi) for i in range(len(yticks))]
for i in range(len(yticks)):
    if yticks[i] == 0:
        ylabels[i] = '0'
plt.yticks(yticks, ylabels)'''
plt.tight_layout()
plt.show()"""

'''for i in range(2, 6):
    plt.plot(times, I[i], marker='.', markersize=3, linewidth=0.7, label='K={:.3g}'.format(K_gen[i]))
plt.title('Intenziteta $I(t)$ za dani $K$')
plt.xlabel('$t$')
plt.ylabel('$I(t)$')
plt.legend()
plt.show()'''

cmap = cmr.bubblegum
#cmap = plt.cm.jet
timesteps = 240000
save_step = 200

theta_out = theta_time_evolution(theta0[ind], K_gen[ind]/gamma1, D=10 * 1e-6, dt=5 * 1e-6, num_timesteps=timesteps, nth_step_save=save_step)
colors = cmap(np.linspace(0, 1, timesteps//save_step))
"""fig, ax = plt.subplots(1, figsize=(6.5,5))
for i in range(len(theta_out)):
    plt.plot(theta_out[i]*180/np.pi, label='K={:.3g}'.format(K_gen[ind]), color=colors[i], zorder=0)
    if i % save_step == 0:
        plt.plot(theta_out[i]*180/np.pi, color='k', lw=0.3, zorder=1)
norm = mpl.colors.Normalize(vmin=0, vmax=T) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
sm.set_array([]) 
cbar = plt.colorbar(sm)
cbar.set_label('$t$', rotation=270, labelpad=16, fontsize=12)
cbarticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
#for j in cbarticks:
#    cbar.ax.plot([0, 1], j*np.ones(2), 'k', lw=0.5)
plt.title('Časovni razvoj profila $\\theta(z,t)$', fontsize = 13)
plt.xlabel('$z$')
plt.ylabel('$\\theta$ [°]')
plt.xticks([0, 99.5, 199], ['$0$', '$d/2$', '$d$'])
plt.xlim(-1, 200)
#plt.yticks(yticks, ylabels)
plt.tight_layout()
plt.show()"""


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4), width_ratios=[6,7])
for i in range(len(theta_out)):
    ax1.plot(theta_out[i]*180/np.pi, label='K={:.3g}'.format(K_gen[ind]), color=colors[i], zorder=0)
    if i % save_step == 0:
        ax1.plot(theta_out[i]*180/np.pi, color='k', lw=0.3, zorder=1)
#ax1.plot(theta_out[0]*180/np.pi, color='r', lw=1.2, zorder=1)
norm = mpl.colors.Normalize(vmin=0, vmax=T) 
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm) 
sm.set_array([]) 
cbar = plt.colorbar(sm)
cbar.set_label('$t$ [s]', rotation=270, labelpad=18, fontsize=None)

points = np.array([times, I[ind]]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap=cmap, norm=norm)
lc.set_array(times)                 #barva intenzitetne krivulje "narašča" s časom
lc.set_linewidth(1.8)
ax2.add_collection(lc)

ax1.set_title('$K={:.2g}$, Časovni razvoj profila $\\theta(z,t)$'.format(K_norm[ind]), fontsize=13)
ax1.set_xlabel('$z$')
ax1.set_ylabel('$\\theta$ [°]')
ax1.set_xticks([0, 99.5, 199], ['$0$', '$d/2$', '$d$'])
ax1.set_xlim(-1, 200)
ax2.set_title('Intenzitetna krivulja $I(t)$', fontsize=13)
ax2.set_xlabel('$t$ [s]')
ax2.set_ylabel('$I(t)$')
ax2.set_xlim(-0.01, T)
ax2.set_ylim(-0.008, 1.008)
#plt.yticks(yticks, ylabels)
plt.tight_layout(w_pad=2)
if noise == 0:
    plt.savefig(abspath + 'intenziteta_{}.png'.format(ind))
else:
    plt.savefig(abspath + 'intenziteta_{}_noise{}.png'.format(ind, noise))
plt.show()