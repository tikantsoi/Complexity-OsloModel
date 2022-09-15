# -*- coding: utf-8 -*-
"""
Document all the results that are found in the report
"""
import numpy as np
import matplotlib.pyplot as plt
import oslo as o
import csv
import pandas as pd
from scipy.optimize import curve_fit
import logbin_2020 as log_bin

#%% Compare the average height at site 1

# L = 16

oslo16 = o.oslo(16, [0.5,0.5])

oslo16.run(100000) 

oslo16.get_average()
# Average height at site 1 = 26.509614132122923

# L = 32

oslo32 = o.oslo(32, [0.5,0.5])

oslo32.run(100000) 

oslo32.get_average()
# Average height at site 1 = 53.86734704167969

#%% Compare the number of recurrent configurations with theoretical values

# L = 4

oslo4 = o.oslo(4, [0.5,0.5])

oslo4.count_reccurent_configurations(34, 1000000)

# Match theoretical value
# There is no new reccurent configuration

#%%

# L = 16, 1000 iterations

oslo16_1000 = o.oslo(16, [0.5,0.5])

oslo16_1000.run(1000) 

"""
We expect to simulate the 1D BTW model if we only allow threshold slope = 1
"""
BTW16_1000 = o.oslo(16, [1,0])

BTW16_1000.run(1000) 

t = np.arange(1,1001)

#%% Compare avalanche size

"""
We expect avlanche size to be always equal to system size for the 1D BTW model 
On the other hand, we expect the avalanche size to grow for the Oslo model
"""

plt.plot(t, oslo16_1000._avalanches, label='Oslo model')
plt.plot(t, BTW16_1000._avalanches, label='BTW model')
plt.xlim(0, 500)
plt.ylim(0, 100)
plt.xlabel("$t$", fontsize=14)
plt.ylabel("$s$", fontsize=14)
plt.legend()

#%% Find probability of the two threhold slopes in the transient configurations

oslo16_1000 = o.oslo(32, [0.5,0.5])

p1, p2 = oslo16_1000.count_threshold_slopes()

print(p1, p2)

# 0.352147577092511 0.6478524229074894 L = 16

# 0.3399822695035461 0.6600177304964538 L = 32

# 0.25594304544994145 0.7440569545500585 L = 512

#%%

BTW16_1000 = o.oslo(16, [1,0])

p1, p2 = BTW16_1000.count_threshold_slopes()

print(p1, p2)

# 0.32211538461538464 0.6778846153846154

#%%

"""
Import the latest data 
"""
oslo4 = pd.read_pickle('files/task3a/oslo4.pkl') 
oslo8 = pd.read_pickle('files/task3a/oslo8.pkl')
oslo16 = pd.read_pickle('files/task3a/oslo16.pkl')
oslo32 = pd.read_pickle('files/task3a/oslo32.pkl')
oslo64 = pd.read_pickle('files/task3a/oslo64.pkl')
oslo128 = pd.read_pickle('files/task3a/oslo128.pkl')
oslo256 = pd.read_pickle('files/task3a/oslo256.pkl')
oslo512 = pd.read_pickle('files/task3a/oslo512.pkl')
oslo1024 = pd.read_pickle('files/task3a/oslo1024.pkl')

#%% Store the heights in a new variable

h4 = oslo4[0][0]
h8 = oslo8[0][0]
h16 = oslo16[0][0]
h32 = oslo32[0][0]
h64 = oslo64[0][0]
h128 = oslo128[0][0]
h256 = oslo256[0][0]
h512 = oslo512[0][0]
h1024 = oslo1024[0][0]

#%%

# plot only up to L = 512

lines = [h4, h8, h16, h32, h64, h128, h256, h512]

for line in lines:
    plt.plot(np.arange(1, len(line) + 1), line)
    plt.xticks(np.arange(0, 1.4e6 + 1, 200000))
    plt.xlabel("$t$", fontsize=14)
    plt.ylabel("$h(t; L)$", fontsize=14)
    plt.legend(['4', '8', '16', '32', '64', '128', '256', '512'], loc=4, title="System size (L)")
    
"""
Curved region represents the transient configurations
Flat region represents the recurrent configurations -> steady state
"""

#%% Calculate the average cross-over time for all system sizes

# include L = 1024

systems = [oslo4, oslo8, oslo16, oslo32, oslo64, oslo128, oslo256, oslo512, oslo1024]

L = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024]) # save as array for easier manipulation

mean_tc = []
mean_std = []

for system in systems:
    mean_tc.append(np.mean(system[-1])) # calculating the mean
    mean_std.append(np.std(system[-1]) / np.sqrt(20))
    
def power(x, a, k):
    return a * x ** k 

popt, pcov = curve_fit(power, L[:-1], mean_tc[:-1])

a = popt[0]

k = popt[1]

sigma_a = np.sqrt(pcov[0][0])

sigma_k = np.sqrt(pcov[1][1])

# a = 0.8448306141929764

# sigma_a = 0.0058165546757960535

# k = 2.002370564949312

# sigma_k = 0.0011113557995094433

#%% Plot average cross-over time vs L

# include L = 1024

plt.errorbar(L, mean_tc, yerr=mean_std, label="Data", fmt='o', capsize=3)
plt.plot(np.arange(4, 1024+1), a*np.arange(4, 1024+1)**2, label="Fit", ls='--')
plt.xlabel('$L$', fontsize=14)
plt.ylabel(r'$\langle t_{c}(L) \rangle$', fontsize=14)
plt.legend(loc=4)

#%% Calculate the mean height (averaged over 20 realisations) for all system sizes

mean_h = []

for system in systems:
    height = 0
    for i in range(20):
        height += pd.Series(system[0][i]) / 20 # array size does not match up, so turn in a series, extra values will be turned into NaN
    mean_h.append(height)

#%% Data collapse plot (log log)

for i in range(len(L)):
    plt.loglog(np.arange(1, len(mean_h[i]) + 1) / L[i]**2, mean_h[i] / L[i])
    plt.legend(L, loc=4, title="System size (L)")
    plt.xlabel('$t$ / $L^{2}$', fontsize=14)
    plt.ylabel(r'$\widetilde{h}(t; L)$ / $L$', fontsize=14)

#%% Zoomed in plot (not log log)

for i in range(len(L)):
    plt.plot(np.arange(1, len(mean_h[i]) + 1) / L[i]**2, mean_h[i] / L[i])
    plt.xlim(0, 5)
    plt.legend(L, loc=4, title="System size (L)")
    plt.xlabel('$t$ / $L^{2}$', fontsize=14)
    plt.ylabel(r'$\widetilde{h}(t; L)$ / $L$', fontsize=14)
    
#%% measure the properties of the plot

# use L = 1024

discontinuous = mean_tc[-1] / L[-1] ** 2

# discontinuity occurs at x = 0.8619845867156982 

sigma_discontinuous = mean_std[-1] / L[-1] ** 2 # standard error

flat = np.mean(mean_h[-1][int(mean_tc[-1]):]) / L[-1]

sigma_flat = np.std(mean_h[-1][int(mean_tc[-1]):]) / len(mean_h[-1][int(mean_tc[-1]):]) / L[-1]

# flat line is found at y = 1.7280466648038701

#%%

"""
We know F(x) follows the power law in the scaling region (x << 1)
Pick the first 500 data points 
Pick L = 512, it is more representative of the data collapse distribution
"""

t_500 = np.arange(1, 501)

popt, pcov = curve_fit(power, t_500 , mean_h[-1][:500] , p0 = [1, 0.5])
    
"""
popt:
array([1.39552028, 0.53617291])

pcov:
array([[ 3.80488600e-05, -4.72229578e-06],
       [-4.72229578e-06,  5.90242233e-07]])

With these parameters, we can reconstruct the exact power law distribution for any system size
"""

a = popt[0]

k = popt[1]

sigma_a = np.sqrt(pcov[0][0])

sigma_k = np.sqrt(pcov[1][1])

#%% plot

plt.errorbar(t_500, mean_h[-1][:500], label="Data", fmt='o', capsize=3, zorder=-1)
plt.plot(t_500, a*t_500**k, '--', lw=2.5,label="Fit")
plt.xlabel('$t$', fontsize=14)
plt.ylabel(r'$\widetilde{h}(t; L)$ ', fontsize=14)
plt.legend(loc=4)


#%%

"""
Calculate the average height, standard deviation and height probability for all system sizes
after they have reached the steady state
"""

av_heights = []
sigmas = []
prob_hs = [] # array
hs = [] # array
sigma_heights = []

for system in systems:
    av_height, sigma, prob_h, h, sigma_height = o.measure(system, 1000000)
    av_heights.append(av_height)
    sigmas.append(sigma)
    prob_hs.append(prob_h)
    hs.append(h)
    sigma_heights.append(sigma_height)
    
#%% Plot average heights vs L

"""
Corrections are not visible yet
"""

plt.scatter(L, av_heights, label="Data")
plt.plot(L, av_heights)
plt.xlabel('$L$', fontsize=14)
plt.ylabel(r'$\langle h \rangle$', fontsize=14)
plt.legend(loc=4)

#%%

"""
Do a curve fit to extract a_0, a_1 and w_1
"""

def corr(L, a0, a1, w1):
    return a0 * L * (1 - a1 * L ** -w1)
    
popt, pcov = curve_fit(corr, L, av_heights)

a0, a1, w1 = popt

"""
popt:
array([1.73392696, 0.24090438, 0.618742  ])

pcov:
array([[ 1.01241176e-07, -3.67849750e-06, -4.52079118e-06],
       [-3.67849750e-06,  1.97122963e-04,  2.06656571e-04],
       [-4.52079118e-06,  2.06656571e-04,  2.31250510e-04]])
"""

sigma_a0 = np.sqrt(pcov[0][0])

sigma_a1 = np.sqrt(pcov[1][1])

sigma_w1 = np.sqrt(pcov[2][2])

#%%

"""
<h> = a_0 * L * (1 - a_1 * L^-w1)
<h> / (a_0 * L) = 1 - a_1 * L^-w1
We shall be able to see the effect of corrections more clearly
Tend to 1 as L -> infinity
"""

#plt.scatter(L, av_heights / (a0 * L), label="Data")
plt.errorbar(L, av_heights / (a0 * L),  yerr=np.array(sigma_heights) / L, label="Data", fmt='o', capsize=3)
plt.plot(L, av_heights  / (a0 * L), '--', label="Fit")
plt.xlabel('$L$', fontsize=14)
plt.ylabel(r'$\langle h(t; L) \rangle_{t}$ / $a_{0} L$', fontsize=14)
plt.legend(loc=4)

#%% Plot standard deviation vs  L

popt, pcov = curve_fit(power, L, sigmas)

"""
popt:
array([0.57324732, 0.24418314])

pcov:
array([[ 8.80886146e-06, -2.60228157e-06],
       [-2.60228157e-06,  8.22064500e-07]])
"""    

a = popt[0]

k = popt[1]

sigma_a = np.sqrt(pcov[0][0])

sigma_k = np.sqrt(pcov[1][1])

plt.errorbar(L, sigmas, label="Data", fmt='o', capsize=3)
plt.plot(np.arange(4,1025), a*np.arange(4,1025)**k, '--', label='Fit')
plt.xlabel('$L$', fontsize=14)
plt.ylabel(r'$\sigma_{h}(L)$', fontsize=14)
plt.legend(loc=4)

#%% log log plot

plt.loglog(L, sigmas, 'o')
plt.loglog(L, sigmas)
plt.xlabel('$L$', fontsize=14)
plt.ylabel(r'$\sigma_{h}$', fontsize=14)

"""
Show a straight line which indicates a power low relationship with L
"""

#%%

"""
h = sum z_i
z_i = <z_i> for L >> 1
h = L * <z> -> <z> = h / L 
<z> found to be 1.73 but should be 1.5
sigma_z = sigma_h / L ~ L^(0.239-1) ~L^-0.761
We exp
"""

#%%

"""
We expect a gaussian distribution that it follows the Central limit theorem
sigma_h should scale with L^-0.5 (this is not the case)
"""

#%% Plot height probability vs h for all system sizes 

"""
Bar graph, bins are centred
"""

for i in range(len(systems)-1):
    plt.bar(hs[i][:-1], prob_hs[i], width=1)
    plt.xlabel('$h$', fontsize=14)
    plt.ylabel('$P$', fontsize=14)
    plt.legend(['4', '8', '16', '32', '64', '128', '256', '512'], loc=1, title="System size (L)")
    
#%%

"""
Continuous line plot
"""



for i in range(5):
    plt.plot(hs[i+2][:-1], prob_hs[i+2])
    plt.xlabel('$h$', fontsize=14)
    plt.ylabel('$P(h; L)$', fontsize=14)
    plt.legend(['16', '32', '64', '128', '256'], loc=1, title="System size (L)")
    
#%% Data collapse plot

"""
We see that sigma_h increases with L and is given by ~ L^0.239
To get rid of the dependence on L and hence to normalize the distribution, it must follow that
we have on the x axis h / sigma_h
To align the peak of all distributions at x = 0, we do h - <h> / L^0.239

On y axis, we must have L^0.239 * P to conserve overall probability
"""

for i in range(9):
    plt.scatter((hs[i][:-1]-av_heights[i]) / (a * L[i]**k), prob_hs[i] * a * L[i]**k)
    plt.xlabel(r'$(h - \langle h \rangle)$ / $\sigma_h$', fontsize=14)
    plt.ylabel('$\sigma_h$ $P(h; L)$', fontsize=14)
    plt.legend(['4', '8', '16', '32', '64', '128', '256', '512', '1024'], loc=1, title="System size (L)")
    
#%%

oslo = o.oslo(1024, [0.5,0.5])

average = oslo.count_slopes(1000000)

# 1.73

#%%

from scipy import stats

x_axis = []

for i in range(9):
    x_axis.append((hs[i][:-1]-av_heights[i]) / (a * L[i]**k))


#%%

x_axis = np.concatenate(x_axis, axis=0)

#%%

skewness = stats.skew(x_axis)

#%%

print(skewness)

#%%
"""
Show scale free behaviour and provides another piece of evidence that there is commality in
the systems

We can calculate the average height and its standard deviation for any system size, as well as
their probabilities

If we sum up all probabilities for every system size, they should sum very close to 1
and the accuracy improves with increasing system size (closer to 1) because they more resemble
the 'true' distribution

We can fit a gaussian curve and read the parameters, they should the same as those of the normal distribution
"""

#%%

"""
The data collapse plot does not follow a Gaussian distribution, it has a long tail on the RHS
(extends to +6) but a short tail on the LHS (extends onlt to -3)
This trend is more obvious with increasing L

We can conclude that zi is not independent and identically distributed
Since we see a bias on the RHS of the average height

Note we should not fit the data collapse plot with a skwed Gaussian distribution because
each system size has a sigma and a skewness, so to summarise their beaviour with only a sigma value is not valid
Instead we can fit individual system size peak with a skewed Gaussian and get their parameters
We shall see their not only the standard deviation scales with L but also the skewness
"""
plt.scatter(hs[0][:-1], prob_hs[0])

#%%

steady_avalanches = []

s_probs = []

ss = []

for system in systems:
    avalanches_steady, s_prob, s = o.process_log_bin_data(system, 1000000)
    steady_avalanches.append(avalanches_steady)
    s_probs.append(s_prob)
    ss.append(s)
    
#%%

centers = []
probs = []

for i in range(len(systems)):
    center, prob = log_bin.logbin(data = steady_avalanches[i], scale = 1.2, zeros = False)
    centers.append(center)
    probs.append(prob)
    
#%% plot thhe data

for i in range(len(systems)):
    plt.loglog(centers[i], probs[i])
    plt.xlabel("$s$", fontsize=14)
    plt.ylabel(r'$\widetilde{P}_{N}(s; L)$', fontsize=14)
    plt.legend(['4', '8', '16', '32', '64', '128', '256', '512', '1024'], loc=3, title="System size (L)")
    
#%% use L=1024 to estimate tau

# first plot its probability function

plt.loglog(centers[-1], probs[-1], '.', ms=2)

#%%

# discounting the first and last 15 points (to be safe), it is clear that it is a straigt line
# this implies a power law relationship

# do a curve fit to these data points

popt, pcov = curve_fit(power, centers[-1][15:-15], probs[-1][15:-15])

a, k = popt

tau = -k

sigma_a = pcov[0][0]

sigma_k = pcov[1][1]

"""
popt:
array([ 0.35645453, -1.54383893])

pcov:
array([[ 4.69020595e-05, -3.29796470e-05],
       [-3.29796470e-05,  2.33412385e-05]])

tau is 1.54383893
"""

#%%

plt.loglog(centers[-1], probs[-1], '.', ms=2, label='Data')
plt.plot(centers[-1][15:-15], a*np.array(centers[-1][15:-15])**k, '--', label='Fit')
plt.xlabel('s', fontsize=14)
plt.ylabel(r'$\widetilde{P}_{N}(s; L)$', fontsize=14)
plt.legend()


#%% now multiply the y axis by s^-tau to align the y dimension

for i in range(len(systems)):
    plt.loglog(centers[i], probs[i]*centers[i]**tau)
    plt.xlabel("$s$", fontsize=14)
    plt.ylabel(r'$s^{\tau_s}$$\widetilde{P}_{N}(s; L)$', fontsize=14)
    plt.legend(['4', '8', '16', '32', '64', '128', '256', '512', '1024'], loc=3, title="System size (L)")

#%% scale the x axis withh L^-D 

# trial and error

# pick out only L = 256 onwards to compare

D_range = np.arange(1, 3, 0.2)

for D in D_range:
    for i in range(3): # last four in the list
        plt.loglog(centers[-i]*L[-i]**-D, probs[-i]*centers[-i]**tau)
        plt.xlabel("$s$ / $L^-{D}$", fontsize=14)
        plt.ylabel(r'$s^{\tau}$$\widetilde{P}$', fontsize=14)
        plt.legend(['256', '512', '1024'], loc=4, title="System size (L)")
    plt.title(f"D = {D}")
    plt.show()

#%% from the trial and error range we can see the data collapse looks best between 
# D = 2.0 and D = 2.3

# try new range

D_range = np.arange(2.1, 2.3, 0.01)

for D in D_range:
    for i in range(3): # last four in the list
        plt.loglog(centers[-i]*L[-i]**-D, probs[-i]*centers[-i]**tau)
        plt.xlabel("$s$ / $L^{D}$", fontsize=14)
        plt.ylabel(r'$s^{\tau_s}$$\widetilde{P}$', fontsize=14)
        plt.legend(['256', '512', '1024'], loc=4, title="System size (L)")
    plt.title(f"D = {D}")
    plt.show()
    
"""
Optimal D we found is 2.15
We know D and tau has to satisfy D(2-tau) = 1
With our D(2-tau) = 0.9807462989803892
D and tau not equal to textbook value because it is subjected to statistical fluctuation in our data?
"""

#%%

D = 2.15

for i in range(9):
    plt.loglog(centers[-i]*L[-i]**-D, probs[-i]*centers[-i]**tau)
    plt.xlabel("$s$ / $L^{D}$", fontsize=14)
    plt.ylabel(r'$s^{\tau_s}$$\widetilde{P}_{N}(s; L)$', fontsize=14)
    plt.legend(['4', '8', '16', '32', '64', '128', '256', '512', '1024'], loc=3, title="System size (L)")
  

#%%

"""
<s^k> is proportional to L^(D(1+k-tau))
"""

# calculate k moment for k = 1, 2, 3, 4 for all system sizes

k_moment_list = []

sigmas = []

for k in range(1,5): # k = 1,2,3,4
    k_moment_system = []
    sigma_system = []
    for system in systems:
        k_moment, sigma = o.find_av_kmoment(system, 1000000, k)
        k_moment_system.append(k_moment)
        sigma_system.append(sigma)
    k_moment_list.append(k_moment_system) # each item in k_moment_list contains all system sizes for a specific k
    sigmas.append(sigma_system)
        
#%%

for i in range(len(k_moment_list)):
    plt.errorbar(L, k_moment_list[i], yerr=sigmas[i], fmt='o', capsize=3)
    plt.xscale("log")
    plt.yscale("log")
    #plt.loglog(L, k_moment_list[i], '.', ms=5)
    plt.xlabel('$L$', fontsize=14)
    plt.ylabel(r'$\langle s^{k} \rangle$', fontsize=14)
    plt.legend(['1', '2', '3', '4'], loc=0, title="k")
    
#%% fit a straight line to measure D(1+k-tau) for each k, this is the exponent

# tried a power law but covariance matrix is very big

# since L >> 1 has to be true, we fit only L = 64 and above

def linear(x, m, c):
    return m*x + c

popt, pcov = curve_fit(linear, np.log(L[3:]), np.log(k_moment_list[0][3:]))

m1 = popt[0]

c1 = popt[1]

sigma_m1 = pcov[0][0]

"""
popt:
array([0.99896094, 0.00426003])

pcov:
array([[ 7.38049541e-08, -3.83684284e-07],
       [-3.83684284e-07,  2.09805930e-06]])
"""

#%%

popt, pcov = curve_fit(linear, np.log(L[3:]), np.log(k_moment_list[1][3:]))

m2 = popt[0] # k = 2

c2 = popt[1]

sigma_m2 = pcov[0][0]

"""
popt:
array([ 3.21562619, -1.26536379])

pcov:
array([[ 7.36679396e-05, -3.82970437e-04],
       [-3.82970437e-04,  2.09414398e-03]])
"""

#%%

popt, pcov = curve_fit(linear, np.log(L[3:]), np.log(k_moment_list[2][3:]))

m3 = popt[0] # k = 2

c3 = popt[1]

sigma_m3 = pcov[0][0]

"""
popt:
array([ 5.45050595, -2.13237485])

pcov:
array([[ 0.00021442, -0.00111467],
       [-0.00111467,  0.00609519]])
"""

#%%

popt, pcov = curve_fit(linear, np.log(L[3:]), np.log(k_moment_list[3][3:]))

m4 = popt[0] # k = 2

c4 = popt[1]

sigma_m4 = pcov[0][0]

"""
popt:
array([ 7.68771973, -2.79811063])

pcov:
array([[ 0.00040552, -0.00210815],
       [-0.00210815,  0.01152768]])
"""

#%%

L_range = np.arange(1, 1025)

for i in range(len(k_moment_list)):
    plt.errorbar(L, k_moment_list[i], yerr=sigmas[i], fmt='o', capsize=3, zorder=-1, label=f'{i+1}')
    plt.legend( loc=0, title="k")
    plt.plot(L_range, L_range**m1*np.exp(c1), '--', color='black', linewidth=1)
    plt.plot(L_range, L_range**m2*np.exp(c2), '--', color='black', linewidth=1)
    plt.plot(L_range, L_range**m3*np.exp(c3), '--', color='black', linewidth=1)
    plt.plot(L_range, L_range**m4*np.exp(c4), '--', color='black', linewidth=1)
    plt.xscale("log")
    plt.yscale("log")
    #plt.loglog(L, k_moment_list[i], '.', ms=5)
    plt.xlabel('$L$', fontsize=14)
    plt.ylabel(r'$\langle s^{k} \rangle$', fontsize=14)
    
    
#%%

m = [m1, m2, m3, m4]

k = np.arange(1, 5)

err = [sigma_m1, sigma_m2, sigma_m3, sigma_m4]

#%%


popt, pcov = curve_fit(linear, k, m)

D = popt[0] # 2.2301156138356055

sigma_D = np.sqrt(pcov[0][0])

c = popt[1] # y intercept, = D - D * tau

sigma_c = np.sqrt(pcov[1][1])

# tau = 1.5547182508405566

"""
popt:
array([ 2.23011561, -1.23708583])

pcov:
array([[ 1.18170058e-05, -2.95425149e-05],
       [-2.95425149e-05,  8.86275461e-05]])

"""    
    
plt.errorbar(k, m, yerr=err, label="Data", fmt='o', capsize=3)

x = np.arange(0, 6)

plt.plot(x, D*x+c, '--', color='darkorange', label='Fit')
plt.legend(loc=0)
plt.xlabel('$k$', fontsize=14)
plt.ylabel(r'$D(1+k-\tau_s)$', fontsize=14)
plt.show()



