#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 09:12:08 2019

@author: maaikevankooten


Simple code snippet to work with the keck data. 
This code does include EOF prediction but I have commented out those sections. 

If you uncomment them DONT just blindly run as it might take 5-10 mins on regular laptop to do determine prediction filter & prediciton on a 60 second dataset etc
Not written to be most optimal but should be slightly clearer to understand. 

1. How to get pseduo open loop phase
2. How to do some EOF on the data. 
3. How to get wavefront error for the pseduo open loop, integrator, and synthetic EOF predictor
4. How to get the temporal PSDs
"""

#code to look at Keck data
import sys
import numpy as np
import matplotlib.pyplot as plt
from hcipy import *
import numpy
import scipy.optimize
import scipy as sp
import astropy
import scipy.signal
import os


def EOF_filter(D,data,alpha=1,flag=0):
    '''
    D: the regressors matrix or the matrix of histroy vectors --> for Keck should be [(m*n),l] vector where m is number of modes, n is the temooral depth of the filter, l is the numer of training sets
    data: is the data vector containing the wavefront measurements delayed by the lag. The shape should be [(m*n),l]
    using the method from Jensen-Clem 2019 for the filter approximation
    '''
    #each wavefront point has a seperate filter so we need to calcualte this all. The regressors stay the same but the data that is give by the filter is differnt. 
    
    F=[]
    idenity=np.eye(D.shape[0])
    print(D.shape)
    print(data.shape)
    print(np.matmul(D,D.transpose()).shape)

    for i in range(data.shape[0]):
        if flag:
           temp1=inverse_truncated(D.transpose(), rcond=1e-4) #need to figet with the rounding condition
           f=np.matmul(temp1,data[i,:].transpose()).transpose()

        else:
            temp1=np.linalg.inv(np.matmul(D,D.transpose())+alpha*idenity)
       # print(temp1.shape)
            temp2=np.matmul(data[i,:],D.transpose())
       # print(temp2.shape)
            f=np.matmul(temp2,temp1)
        
        F.append(f)
#    print(temp1.shape)
#    print(temp2.shape)
#    print(f.shape)
    return np.array(F)
def temporal_PSD(data,Hz):
    #assume that time in the 2-dimension
    PSD=[]
    nblock=1024*2
    overlap=nblock*0.25
    win=scipy.signal.hanning(nblock,True)
    f=0
    print(data.shape)
    for k in range(data.shape[0]):
        use_data=data[k,:]
        n,x=np.isnan(use_data), lambda z: z.nonzero()[0]
        use_data[n]=np.interp(x(n),x(~n),use_data[~n])
        use_data[use_data>0]=np.mean(use_data)
        f,Pxx=scipy.signal.welch(use_data,window=win, noverlap=overlap, nfft=nblock,fs=Hz,detrend=False,scaling='density')
        PSD.append(Pxx)
    return np.mean(PSD,axis=0),f

def plot_results(foldername,file_list,labels=['LMMSE'],length=int(120*1000)):
   #   The RMS wavefront error is given by a square root of the difference between 
  #  the average of squared wavefront deviations minus the square of average wavefront deviation,
  #  <W2>-<W>2
    c=['#a50026',"#d73027","#fc8d59","#fee090","#91bfdb","#4575b4",'#313695']

    plt.figure()
    for i in range(len(file_list)):
        data=numpy.load(foldername+file_list[i])
        full_rms=data['arr_0']
        integrator_rms=data['arr_1']
        prediction_rms=data['arr_2']
        time=np.linspace(0,int(length/1000),length)
        if i==0:
                plt.semilogy(time,np.array(full_rms),'k',label='Full phase')
                plt.semilogy(time,np.array(integrator_rms),'b',label='Keck residuals - no prediction')
        print(labels[i])
        plt.semilogy(time,np.array(prediction_rms),label=labels[i],color=c[i])
        
    plt.ylabel('RMS wfe [um]', fontsize=15)
    plt.xlabel('Seconds [s]', fontsize=15)            
    plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=int((len(labels)+2)/2),fancybox=True,prop={'size':10})
    return   
def pupil(N):

    p = np.zeros([N,N])

    radius = N/2.
    [X,Y] = np.meshgrid(np.linspace(-(N-1)/2.,(N-1)/2.,N),np.linspace(-(N-1)/2.,(N-1)/2.,N))
    R = np.sqrt(pow(X,2)+pow(Y,2))
    p[R<=radius] = 1
    
    return p
folder='/Users/maaikevankooten/Documents/work/Data/Keck_telemetry/first_keck/'
#########3#So, to get the pseudo open loop phase values, I do the following:##########
pup=pupil(21)
DM_commands = np.load(folder+'/dmc.npy')[:,pup>0]
DM_residual = np.load(folder+'./residualWF.npy')[:,pup>0]

DM_commands[DM_commands>10] = 10
DM_commands[DM_commands<-10] = -10

open_loop = (-1*DM_commands + DM_residual)
open_loop_phase = open_loop * 0.6
simple_integrator=DM_residual*0.6


#################Lets setup some EOF######################
# from the data lets create our regressor dataset and our input data set
Hz=1E3 #the running freqeuncy of the WFS
# n=5 #temporal depth of the filter solution
# l=15*Hz #length of the training set 
# delay=1 #the assumed delay in the AO loop
# m=open_loop_phase.shape[1]
# regressors=[]
# data=[]
# old_data=[] #list containing the regressors
# for kkk in range(n):
#     old_data.append(np.zeros(int(m)))   

# for i in range(int(l)):
#     regressors.append(np.asarray(old_data).ravel()) #I start at the nth sample so that I dont need to deal with zeros in the regressors to start
#     old_data.pop(0)
#     if i<(delay-1):
#         old_data.append(np.zeros(int(m))) #append at the end/bottom of the list
#     else:
#         old_data.append(open_loop_phase[i-(delay-1),:]) #append at the end/bottom of the list
#     data.append(open_loop_phase[i,:].ravel())
# regressors=np.array(regressors)
# data=np.array(data)

#now we can use this to train the data set and get our filter solution
#F=EOF_filter(regressors.transpose(),data.transpose(),flag=1)

#now lets do some prediction
# samples=10*Hz-l #10 seconds of data to start with
# old_data=[] #list containing the regressors
# for kkk in range(n):
#     old_data.append(np.zeros(int(m)))
residual_phase=[]
# count=0
# for mmm in range(0,int(samples+l)):
    
#     if mmm<int(l):
#         predicted_value=np.zeros((349,))
#     else:
#         predicted_value=np.matmul(F,np.asarray(old_data).ravel())

#     residual_phase.append(open_loop_phase[mmm,:]-predicted_value)
#     old_data.pop(0) #pop of the top of the listwell t
    
#     if mmm<(delay-1):
#         old_data.append(np.zeros(int(m))) #append at the end/bottom of the list
#     else:
#         old_data.append(open_loop_phase[mmm-(delay-1),:]) #append at the end/bottom of the list
#     count+=1

#T###########he RMS wavefront error is given by a square root of the difference between 
#the average of squared wavefront deviations minus the square of average wavefront deviation,
#<W2>-<W>2
prediction_rms=[]
full_rms=[]
integrator_rms=[]
for kkk in range(len(simple_integrator)):
  #  prediction_rms.append(np.sqrt(np.mean(residual_phase[kkk]**2)-np.mean(residual_phase[kkk])**2))
    full_rms.append(np.sqrt(np.mean(open_loop_phase[kkk]**2)-np.mean(open_loop_phase[kkk])**2))
    integrator_rms.append(np.sqrt(np.mean(simple_integrator[kkk]**2)-np.mean(simple_integrator[kkk])**2)) 

plt.figure(figsize=(12,7))
time=np.linspace(0,int(len(full_rms)/Hz),len(full_rms))
plt.semilogy(time,np.array(full_rms),label='Full phase')
plt.semilogy(time,np.array(integrator_rms),label='Integrator residual phase')
#plt.semilogy(time,np.array(prediction_rms),label='Prediction residual phase')
plt.ylabel('RMS wfe [um]', fontsize=15)
plt.xlabel('Seconds [s]', fontsize=15)

plt.legend(loc='upper center',bbox_to_anchor=(0.5,1.15),ncol=4,fancybox=True,prop={'size':12})
#np.savez(folder+'training_rms',full_rms,integrator_rms,prediction_rms)
np.savez(folder+'training_phase',np.asarray(open_loop_phase),np.asarray(simple_integrator),np.asarray(residual_phase))

 #####okay lets calulate some temporal PSD's 
 
data=np.load(folder+'training_phase.npz')
open_loop_phase=data['arr_0']
simple_integrator=data['arr_1']

full_temp_psd,f_full=temporal_PSD(np.asarray(open_loop_phase)[int(l):int(samples+l),:].transpose(), Hz)
int_temp_psd,f_int=temporal_PSD(np.asarray(simple_integrator)[int(l):int(samples+l),:].transpose(), Hz)

#residual_phase=data['arr_2'] #prediction; comment this and the next line out
#pre_temp_psd,f_pre=temporal_PSD(np.asarray(residual_phase)[int(l):int(samples+l),:].transpose(), Hz)


plt.figure()
plt.loglog(f_full,full_temp_psd,label='Full')
#plt.loglog(f_pre,pre_temp_psd,label='Pred')
plt.loglog(f_int,int_temp_psd,label='Integrator')
plt.ylabel('PSD')
plt.xlabel('Freq[Hz]')
plt.legend()


