import matplotlib.pyplot as plt
import numpy as np
import copy
import time
import os
import tensorflow as tf
import keras
from keras import Sequential
from keras.layers import Dense
colors = plt.cm.Dark2([0, 0.2, 0.3])

#-----------------SETTING ENVIROMENT---------------------------------------------------------------------------------------------
OKOLJE = 0      #0 - osebni računalnik, 1 - Marvin
if OKOLJE == 0:
    abspath = r"C:\FMF\Magisterij\Praktikum_strojnega_ucenja\PSUF_naloga3\\"
    savepath = r"C:\FMF\Magisterij\Praktikum_strojnega_ucenja\PSUF_naloga3\\"
else:
    savepath = r"/home/jaksetk/psuf_naloga3/"
    datapath = r"/data/PSUF_naloge/3-naloga/DataK/"
#--------------------------------------------------------------------------------------------------------------------------------

#-----------------LOADING DATA---------------------------------------------------------------------------------------------------
lamb = 505      #valovna dolžina [nm]
noise = 0       #šum
if OKOLJE == 0:
    #I = np.load(os.path.join(abspath + 'pregenerated_data', 'intensity{}noise{}.npy'.format(lamb, noise)))      #[:,::10]
    I = np.load("C:\FMF\Magisterij\Praktikum_strojnega_ucenja\PSUF_naloga3\intensity{}noise{}.npy".format(lamb, noise))
    K_gen = np.load(os.path.join(abspath + 'pregenerated_data', 'Kvalues.npy'))
else:
    I = np.load(os.path.join(datapath, 'intensity{}noise{}.npy'.format(lamb, noise)))       #[:,::10]
    K_gen = np.load(os.path.join(datapath, 'Kvalues.npy'))
Kmax = 20e-12
K_gen /= Kmax
N = len(I)      #število parov I(t) - K

#times = np.linspace(0, len(I[0])-1, len(I[0]))    #indeksi časov (default: [0, 399])
T = 1.2     #končni čas [s]
nt = len(I[0])      #število časovnih točk
print(nt)
times = np.linspace(0, T, nt)

#--------------------------------------------------------------------------------------------------------------------------------

#plt.plot([i for i in range(1, len(I[0])+1)], I[0])
'''for i in range(11):
    plt.plot(times, I[i], marker='.', markersize=3, linewidth=0.7, label='K={:.3f}'.format(K_gen[i]))
plt.title('Intenziteta $I(t)$ za dani $K$')
plt.xlabel('$t$')
plt.ylabel('$I(t)$')
plt.legend()
plt.show()'''

#-----------------setting TEST, VALIDATION and TEST DATA-------------------------------------------------------------------------
X = copy.copy(I)
Y = copy.copy(K_gen)

N_train = int(0.8*N)                    #velikost train (trenirnega) seta
N_val = int((N-N_train)/2)              #velikost validacijskega seta
N_test = N - N_train - N_val            #velikost testnega seta

X_train = X[:N_train]
X_val = X[N_train:N_train + N_val]
X_test = X[N_train + N_val:]

Y_train = Y[:N_train]
Y_val = Y[N_train:N_train + N_val]
Y_test = Y[N_train + N_val:]

'''print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)'''
#--------------------------------------------------------------------------------------------------------------------------------

#---------------------------NEVRONSKA MREŽA------------------------------------------------------------------------------------------------
NE = 100         #number of epochs
#BS = 100         #batch size
lr = 5e-4       #learning rate

'''time1 = time.time()
model = Sequential()
model.add(Dense(200, activation='relu', input_shape=(nt,)))              #default: nt = 400
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_absolute_error', metrics = ['mean_squared_error'])
fit1 = model.fit(X_train, Y_train, epochs=NE, batch_size=BS, validation_data=(X_val,Y_val), shuffle=True)
time2 = time.time()'''
#------------------------------------------------------------------------------------------------------------------------------------------

#arch=[nt, 200, 100, 1]         #osnovni model
arch=[nt, 100, 50, 1]           #tega sem uporabil v primeru ko smo vzeli I(t) samo v vsaki drugi točki
activ='relu'
def nevronska(arch, activ, NE, BS, lr):         #architecture=[nt,...,1], Activation Function, Number of Epochs, Batch Size, Learning Rate
    model = Sequential()
    model.add(Dense(arch[1], activation=activ, input_shape=(nt,)))              #default: nt = 400
    for i in range(2, len(arch)):
        model.add(Dense(arch[i], activation=activ))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mean_absolute_error', metrics = ['mean_squared_error'])
    fit1 = model.fit(X_train, Y_train, epochs=NE, batch_size=BS, validation_data=(X_val,Y_val), shuffle=True)
    return model, fit1

#modeli pri številu epoh NE = 100 in različne velikosti batchev BS:

#batch_sizes = np.array([5, 10, 25, 50, 80, 100])
batch_sizes = np.array([200])

"""comp_times = np.empty(len(batch_sizes))
for i in range(len(batch_sizes)):
    while True:
        time7 = time.time()
        model, fit1 = nevronska(arch, activ, NE, batch_sizes[i], lr)
        time8 = time.time()
        if fit1.history['val_loss'][int(NE/2)] != fit1.history['val_loss'][-1]:     #če fit ne skonvergira, poskuša naprej
            break

    comp_times[i] = time8 - time7

    np.save(savepath + 'history_arch{}_NE{}_BS{}_lr{}.npy'.format(arch, NE, batch_sizes[i], lr), fit1.history)
    model.save(savepath + 'saved_models/model_arch{}_NE{}_BS{}_lr{}.keras'.format(arch, NE, batch_sizes[i], lr))

    Y_pred = model.predict(X_test)[:,0]

    plt.figure(figsize=(5,5))
    plt.plot(Y_test, Y_pred, 'k.', markersize=1, alpha=0.5)
    plt.xlabel('$K_{true}$')
    plt.ylabel('$K_{pred.}$')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.title('Napovedi: $\lambda = {}$ nm, šum = {}'.format(lamb, noise))
    plt.tight_layout()
    if OKOLJE == 0:
        plt.savefig(abspath + 'evaluacija.png')
        plt.show()
    else:
        plt.savefig(savepath + 'evaluacija_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, batch_sizes[i], lr))

    '''plt.figure(figsize=(6,3))
    plt.plot(np.linspace(1, NE, NE), fit1.history['loss'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
    plt.plot(np.linspace(1, NE, NE), fit1.history['val_loss'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
    plt.xlabel('Epohe')
    plt.ylabel('MAE')
    plt.xlim(0, NE+1)
    #plt.yscale('log')
    plt.legend()
    plt.title('Funkcija izgube (MAE) za NE={}, BS={}, lr={}'.format(NE, batch_sizes[i], lr))
    plt.tight_layout()
    if OKOLJE == 0:
        plt.savefig(abspath + 'MAE.png')
        plt.show()
    else:
        plt.savefig(savepath + 'MAE2_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, batch_sizes[i], lr))

    plt.figure(figsize=(6,3))
    plt.plot(np.linspace(1, NE, NE), fit1.history['mean_squared_error'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
    plt.plot(np.linspace(1, NE, NE), fit1.history['val_mean_squared_error'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
    plt.xlabel('Epohe')
    plt.ylabel('MSE')
    plt.xlim(0, NE+1)
    #plt.yscale('log')
    plt.legend()
    plt.title('Metrika (MSE) za NE={}, BS={}, lr={}'.format(NE, batch_sizes[i], lr))
    plt.tight_layout()
    if OKOLJE == 0:
        plt.savefig(abspath + 'MSE.png')
        plt.show()
    else:
        plt.savefig(savepath + 'MSE2_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, batch_sizes[i], lr))'''

    #------------plot rezultatov modela--------------------
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

    ax1.plot(fit1.history['loss'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
    ax1.plot(fit1.history['val_loss'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
    ax1.set_xlabel('Epohe', fontsize=12)
    ax1.set_ylabel('MAE', fontsize=12)
    ax1.set_xlim(-1, NE)
    ax1.set_ylim(0.02, 0.12)           #specific setting...from how the results were
    ax1.legend()
    ax1.set_title('Funkcija izgube (MAE)', fontsize=13)

    ax2.plot(fit1.history['mean_squared_error'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
    ax2.plot(fit1.history['val_mean_squared_error'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
    ax2.set_xlabel('Epohe', fontsize=12)
    ax2.set_ylabel('MSE', fontsize=12)
    ax2.set_xlim(-1, NE)
    ax2.set_ylim(0, 0.03)           #specific setting...from how the results were
    ax2.legend()
    ax2.set_title('Metrika (MSE)', fontsize=13)
    plt.tight_layout()
    plt.savefig(savepath + 'resultsA_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, batch_sizes[i], lr))


    fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True)
    my_cmap = copy.copy(plt.cm.inferno_r)
    my_cmap.set_under('lightskyblue', 1)
    hg = ax3.hist2d(Y_test, Y_pred, bins=100, range=[[0,1],[0,1]], cmap=my_cmap, vmin=1)
    ax3.set_xlabel('$K_{true}$', fontsize=13)
    ax3.set_ylabel('$K_{pred.}$', fontsize=13)
    ax3.tick_params(left=True, right=True)
    ax3.set_aspect('equal')
    fig.colorbar(hg[3], fraction=0.046, ax=ax3)
    plt.tight_layout()
    plt.savefig(savepath + 'resultsB_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, batch_sizes[i], lr))"""
    
    #------------------------------------------------------

'''plt.figure(figsize=(5,5))
plt.plot(batch_sizes, comp_times, color='b', marker='.', lw=0.8)
plt.xlabel('Batch Size')
plt.ylabel('$t$ [s]')
plt.xlim(0, NE*1.05 + 1)
plt.ylim(0, None)
plt.title('Računski čas za {} epoh ($\lambda = {}$ nm, šum = {})'.format(NE, lamb, noise))
plt.tight_layout()
if OKOLJE == 0:
    plt.savefig(abspath + 'evaluacija.png')
    plt.show()
else:
    plt.savefig(savepath + 'comptimesBS_arch{}_NE{}_lr{}.png'.format(arch, NE, lr))
print(comp_times)
np.save(savepath + 'comptimesBS_arch{}_NE{}_lr{}.npy'.format(arch, NE, lr), comp_times)

colors2 = iter([plt.cm.Dark2(i) for i in range(8)])
plt.figure(figsize=(5,5))
plt.xlabel('Epohe')
plt.ylabel('MAE')
plt.xlim(0, NE+1)
plt.title('Funkcija izgube za {} epoh ($\lambda = {}$ nm, šum = {})'.format(NE, lamb, noise))
for i in range(len(batch_sizes)):
    fit_i = np.load(savepath + 'history_arch{}_NE{}_BS{}_lr{}.npy'.format(arch, NE, batch_sizes[i], lr), allow_pickle='TRUE').item()
    plt.plot(np.linspace(1, NE, NE), fit_i['loss'], marker='.', ms=4, lw=0.8, label='BS = {}'.format(batch_sizes[i]), c=next(colors2))
plt.legend()
plt.tight_layout()
if OKOLJE == 0:
    plt.savefig(abspath + 'evaluacija.png')
    plt.show()
else:
    plt.savefig(savepath + 'lossBS_arch{}_NE{}_lr{}.png'.format(arch, NE, lr))'''

    
#SAVING MODEL
'''if OKOLJE == 0:
    model.save(abspath + 'saved_models/model2.keras')
else:
    np.save(savepath + 'history_arch{}_NE{}_BS{}_lr{}.npy'.format(arch, NE, BS, lr), fit1.history)
    model.save(savepath + 'saved_models/model_arch{}_NE{}_BS{}_lr{}.keras')'''

#LOADING MODEL
"""BS = 5             #batch_sizes = np.array([5, 10, 25, 50, 80, 100]) ...za te vrednosti smo fitali nevronsko mrežo
if OKOLJE == 0:
    model = keras.models.load_model(abspath + 'saved_models/model2.keras')
else:
    model = keras.models.load_model(savepath + 'saved_models/model_arch{}_NE{}_BS{}_lr{}.keras'.format(arch, NE, BS, lr))
    fit1 = np.load(savepath + 'history_arch{}_NE{}_BS{}_lr{}.npy'.format(arch, NE, BS, lr), allow_pickle='TRUE').item()

time3 = time.time()
Y_pred = model.predict(X_test)[:,0]
np.save(savepath + 'K_pred_NE{}_BS{}_lr{}.npy'.format(NE, BS, lr), Y_pred)
#Y_pred = np.load(savepath + 'K_pred_NE{}_BS{}_lr{}.npy'.format(NE, BS, lr))
time4 = time.time()

#time5 = time.time()
'''plt.figure(figsize=(5,5))
plt.plot(Y_test, Y_pred, 'k.', markersize=1, alpha=0.5)
plt.xlabel('$K_{true}$')
plt.ylabel('$K_{pred.}$')
plt.xlim(0,1)
plt.ylim(0,1)
plt.title('Napovedi: $\lambda = {}$ nm, šum = {}'.format(lamb, noise))
plt.tight_layout()
if OKOLJE == 0:
    plt.savefig(abspath + 'evaluacija.png')
    plt.show()
else:
    plt.savefig(savepath + 'evaluacija_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))'''
#time6 = time.time()
###print('fitanje: {} s'.format(time2-time1))
#print('napovedovanje: {} s'.format(time4-time3))
#print('plotanje: {} s'.format(time6-time5))

my_cmap = copy.copy(plt.cm.inferno_r)
my_cmap.set_under('lightskyblue', 1)

#2D histogram
plt.figure(figsize=(6,5))
plt.hist2d(Y_test, Y_pred, bins=100, range=[[0,1],[0,1]], cmap=my_cmap, vmin=1)
plt.xlabel('$K_{true}$')
plt.ylabel('$K_{pred.}$')
plt.colorbar()
plt.tight_layout()
plt.savefig(savepath + 'eval_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))       #eval kot evaluacija...2d histogram predicted vs true

colors = plt.cm.Dark2([0, 0.2, 0.3])

#history1 = copy.copy(fit1.history)          #for a new calculation
history1 = copy.copy(fit1)                  #when loading data

'''plt.figure(figsize=(6,3))
plt.plot(history1['loss'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
plt.plot(history1['val_loss'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
plt.xlabel('Epohe')
plt.ylabel('MAE')
plt.xlim(0, NE)
#plt.yscale('log')
plt.legend()
plt.title('Funkcija izgube (MAE)')
plt.tight_layout()
if OKOLJE == 0:
    plt.savefig(abspath + 'MAE.png')
    plt.show()
else:
    plt.savefig(savepath + 'MAE_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))

plt.figure(figsize=(6,3))
plt.plot(history1['mean_squared_error'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
plt.plot(history1['val_mean_squared_error'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
plt.xlabel('Epohe')
plt.ylabel('MSE')
plt.xlim(0, NE)
#plt.yscale('log')
plt.legend()
plt.title('Metrika (MSE)')
plt.tight_layout()
if OKOLJE == 0:
    plt.savefig(abspath + 'MSE.png')
    plt.show()
else:
    plt.savefig(savepath + 'MSE_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))'''

#-------plot rezultatov modela---------
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 6), sharex=True)

ax1.plot(history1['loss'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
ax1.plot(history1['val_loss'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
ax1.set_xlabel('Epohe', fontsize=12)
ax1.set_ylabel('MAE', fontsize=12)
ax1.set_xlim(-1, NE)
ax1.set_ylim(0.02, 0.12)           #specific setting...from how the results were
ax1.legend()
ax1.set_title('Funkcija izgube (MAE)', fontsize=13)

ax2.plot(history1['mean_squared_error'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
ax2.plot(history1['val_mean_squared_error'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
ax2.set_xlabel('Epohe', fontsize=12)
ax2.set_ylabel('MSE', fontsize=12)
ax2.set_xlim(-1, NE)
ax2.set_ylim(0.005, 0.03)           #specific setting...from how the results were
ax2.legend()
ax2.set_title('Metrika (MSE)', fontsize=13)
plt.tight_layout()
plt.savefig(savepath + 'resultsA_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))


fig, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), sharex=True)
my_cmap = copy.copy(plt.cm.inferno_r)
my_cmap.set_under('lightskyblue', 1)
hg = ax3.hist2d(Y_test, Y_pred, bins=100, range=[[0,1],[0,1]], cmap=my_cmap, vmin=1)
ax3.set_xlabel('$K_{true}$', fontsize=13)
ax3.set_ylabel('$K_{pred.}$', fontsize=13)
ax3.tick_params(left=True, right=True)
ax3.set_aspect('equal')
fig.colorbar(hg[3], fraction=0.046, ax=ax3)
plt.tight_layout()
plt.savefig(savepath + 'resultsB_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))


#še vse skupaj (žal premajhno za v poročilo):
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16,4.5))
  
ax1.plot(history1['loss'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
ax1.plot(history1['val_loss'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
ax1.set_xlabel('Epohe')
ax1.set_ylabel('MAE')
ax1.set_xlim(0, NE)
ax1.legend()
ax1.set_title('Funkcija izgube (MAE)')

ax2.plot(history1['mean_squared_error'], marker='.', label='učni podatki', color='mediumblue', lw=0.8)
ax2.plot(history1['val_mean_squared_error'], marker='.', label='validacijski podatki', color=colors[1], lw=0.8)
ax2.set_xlabel('Epohe')
ax2.set_ylabel('MSE')
ax2.set_xlim(0, NE)
ax2.legend()
ax2.set_title('Metrika (MSE)')

my_cmap = copy.copy(plt.cm.inferno_r)
my_cmap.set_under('lightskyblue', 1)
hg = ax3.hist2d(Y_test, Y_pred, bins=100, range=[[0,1],[0,1]], cmap=my_cmap, vmin=1)
ax3.set_xlabel('$K_{true}$')
ax3.set_ylabel('$K_{pred.}$')
ax3.set_aspect('equal')
ax3.set_title('Napoved vs resnica')
fig.colorbar(hg[3], fraction=0.046, ax=ax3)
#plt.suptitle('Rezultati modela: arch={}, $N_e={}$, BS={}, lr={}'.format(arch, NE, BS, lr), fontsize=13)
plt.tight_layout()
plt.savefig(savepath + 'results_arch{}_NE{}_BS{}_lr{}.png'.format(arch, NE, BS, lr))
#--------------------------------------
"""

#---------------- modeli trenirani na čistih podatkih, uporabljeni na zašumljenih --------------------------------
#na vrhu kode izberi zašumljene podatke!

arch=[nt, 200, 100, 1]
#batch_sizes = np.array([5, 10, 25, 50, 80, 100])
#noise = 0
my_cmap = copy.copy(plt.cm.inferno_r)
my_cmap.set_under('lightskyblue', 1)

for i in range(len(batch_sizes)):
    model = keras.models.load_model(savepath + 'saved_models\model_arch{}_NE{}_BS{}_lr{}.keras'.format(arch, NE, batch_sizes[i], lr))
    #fit1 = np.load(savepath + 'history_arch{}_NE{}_BS{}_lr{}.npy'.format(arch, NE, BS, lr), allow_pickle='TRUE').item()
    #model = keras.models.load_model(savepath + 'saved_models/model_arch{}_NE{}_BS{}_lr{}.keras'.format(arch, NE, batch_sizes[i], lr))

    Y_pred = model.predict(X_test)[:,0]

    #2D histogram
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    ax.hist2d(Y_test, Y_pred, bins=100, range=[[0,1],[0,1]], cmap=my_cmap, vmin=1)
    ax.set_xlabel('$K_{true}$')
    ax.set_ylabel('$K_{pred.}$')
    ax.set_aspect('equal')
    plt.colorbar()
    plt.title('Čisti model na zašumljenih podatkih, $\lambda = {}$ nm, '.format(lamb))
    plt.tight_layout()
    #plt.savefig(savepath + 'eval_arch{}_NE{}_BS{}_lr{}_CleanModel_DirtyData.png'.format(arch, NE, batch_sizes[i], lr))       #eval kot evaluacija...2d histogram predicted vs true
    #plt.savefig(savepath + 'poskusss.png')

#----------------------------------------------------------------------------------------------------------------------



'''model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_I=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
)'''
