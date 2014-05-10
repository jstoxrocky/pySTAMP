# -*- coding: utf-8 -*-
import numpy as np

from pandas import DataFrame
import pandas as pd

import scipy as sp
import scipy.stats
#from scipy.optimize import minimize
from _minimize_joe import *

from other_functions import *
from initialize import init_em

class sts:

    def __init__(self,y,comp_dic,n_forecast = 0):

        self.c_count = 0
        s = 1
        #get number of (diffuse) components (state vars)
        #irr not counted as state var
        n_diff_states = 0
        if "level" in comp_dic: #adds 1 to the state vector
            n_diff_states += 1
        if "slope" in comp_dic: #adds 1 to the state vector
            n_diff_states += 1
        if "cycle" in comp_dic: #adds 2 to the state vector
            n_diff_states += 2
        if "seasonal" in comp_dic: #adds s-1 to the state vector
            s = comp_dic["seasonal"][1]
            n_diff_states += s - 1


        
        #create a fixed list and variable list of variables
        fix_list = []
        var_list = []
        for d in comp_dic:

            if d == "seasonal":
                if comp_dic[d][0] == "var":
                    var_list.append(d)
                elif comp_dic[d][0] == "fix":
                    fix_list.append(d)
            else:
                if comp_dic[d] == "var":
                    var_list.append(d)
                elif comp_dic[d] == "fix":
                    fix_list.append(d)
                #this else is for cycles and seasonal components, since they take integer values of variable
                else:
                    var_list.append(d)
            


        #for now we are assuming cycle adds 1 estimated variance (is this a good assumption?)
        n_disturbance = len(var_list)


        #create static matrices
        Z = np.zeros(n_diff_states)
        R = np.matrix(np.identity(n_diff_states))

        T = np.matrix(np.identity(n_diff_states - (s-1))) #add to concat laters

        try:
            #for seasonal
            b = np.matrix(np.identity(s - 1))
            b = b[:-1]

            t = [-1]*(s - 1)
            t = np.matrix(t)
            Ts = np.concatenate((t, b), axis=0)
            T = concat(T, Ts)
        except:
            pass



        #modify T matrix if slope term in model specs
        if "slope" in comp_dic and "level" in comp_dic: 
            T[0,1] = 1    

        #we assume the order of components goes
        #level
        #slope
        #cycle
        #seasonal

        count = 0
        if "level" in comp_dic:
            zpos_level = count
            Z[zpos_level] = 1
            count += 1
        if "slope" in comp_dic:
            zpos_slope = count
            count += 1
        if "cycle" in comp_dic:
            zpos_cycle = count
            Z[zpos_cycle] = 1
            count += 2
        if "seasonal" in comp_dic:
            zpos_seasonal = count
            Z[zpos_seasonal] = 1
            count += s - 1

        Z = np.matrix(Z)        

        #for dataframe organization
        n = len(y)
        y_f = y
        for i in xrange(n_forecast):
            y_f = y_f.set_value(n+i, np.nan)

        #export global variables
        self.s = s
        self.y = y
        #self.y_f = y_f
        self.n = len(y)
        self.comp_dic = comp_dic
        self.n_diff_states = n_diff_states

        #self.filter_state = DataFrame(y) #where is this used?
        #self.smooth_state = DataFrame(y) #where is this used?
        #self.smooth_state2 = DataFrame(y) #where is this used?

        self.fsd = {"y":y}
        self.ss2d = {"y":y}

        self.Z = Z
        self.R = R
        self.T = T
        self.D = np.matrix([[0]])
        self.C = np.matrix([[0]])
        self.fix_list = fix_list
        self.var_list = var_list
        self.n_disturbance = n_disturbance
        self.n_forecast = n_forecast



        #Allow Kalman filtering function callable
        #could this be sent to this class as an object??
        #This function needed to be here to populate the k filter with the self variables
        #and also to call and run the k filter
        def calc_kalman(x0, flag):

            #import global variables 
            y = self.y
            n = self.n
            Z = self.Z
            D = self.D
            C = self.C
            R = self.R
            T = self.T
            n_diff_states = self.n_diff_states
            n_disturbance = self.n_disturbance
            comp_dic = self.comp_dic
            conc_param = self.conc_param
            var_list = self.var_list
            s = self.s

            #assign parameter values to H and Q
            #and initialize A and P
            H,Q,A,P,T = set_param_vals(comp_dic,conc_param,var_list,flag,x0,n_diff_states,y,T,s)
            
            #apply kalman filter
            #if final run, calc with forecast
            if flag == 2:
                LL, LLc, dVar, V, F, L, EA, EP, K, H, Q, R, A, P, C, sig_rec, KA, KP, KV, KF, KK, KL = k_filter(n,T,A,C,P,R,Q,y,Z,D,H,n_diff_states,n_disturbance,n_forecast)
            #during optimisation do not use forcast
            else:
                LL, LLc, dVar, V, F, L, EA, EP, K, H, Q, R, A, P, C, sig_rec, KA, KP, KV, KF, KK, KL = k_filter(n,T,A,C,P,R,Q,y,Z,D,H,n_diff_states,n_disturbance)


            #flag = 0
            #EM initialization step
            if flag == 0:
                
                #make variables available to smoothing algorithm
                self.KA = KA
                self.KP = KP
                self.KV = KV
                self.KF = KF
                self.KK = KK
                self.KL = KL
                
                self.V = V
                self.F = F
                self.L = L
                self.K = K
                self.H = H
                self.Q = Q
                self.R = R
                self.A = A
                self.P = P
                self.C = C
                self.Z = Z
                self.T = T
                self.EA = EA
                self.EP = EP
                self.sig_rec = sig_rec
                self.n_diff_states = n_diff_states
                self.flag = 0
                
                #return paramters from EM algorithm
                EM_x0 = self.smooth()
                
                return EM_x0, LL, dVar
            
            #flag = 1
            #flag indicating that parameters are still being calibrated
            elif flag == 1:

                print(np.exp(x0))
                
                #if no/yes concentrated paramter return correct loglikelihood 
                if conc_param == "none":
                    return -LL
                else:
                    return -LLc
            
            #flag = 2
            #parameters have been evaluated and this is last run of function
            elif flag == 2:
                
                #if no/yes concentrated paramter return correct loglikelihood 
                if conc_param == "none":
                    my_LL = -LL
                else:
                    my_LL =  -LLc
                
                return my_LL, A, EA, P, EP, T, C, V, F, L, R, Q, K, H, sig_rec, KA, KP, KV, KF, KK, KL
         
                      
        #set parameters to values from STAMP
        x0 = []
        if "irr" in comp_dic:
            x0.append(-1.0)
        if "level" in comp_dic:
            x0.append(-0.5)
        if "slope" in comp_dic:
            x0.append(-1.5)
        if "cycle" in comp_dic:
            if comp_dic["cycle"] != "fix":
                x0.append(-1.5)
                damp = 2.0
                period = comp_dic["cycle"]
                freq = 2.0*np.pi/float(comp_dic["cycle"])
            else:
                x0.append(0.0)
                damp = 1.0
                period = 14.0
                freq = 2.0*np.pi/14.0
        if "seasonal" in comp_dic:
            x0.append(-2.0)

        #no concentrate parameter (for now)
        self.conc_param = "none"

        #initialize with EM algorthm
        init_list, conc_param = init_em(calc_kalman,x0,comp_dic)
        
        #export variables
        self.init_list = init_list  
        self.conc_param = conc_param
         
        #make kalman function accesible to rest of class methods
        self.kalman = calc_kalman
    





    def filter(self):
        
        #import global variables
        comp_dic = self.comp_dic
        n = self.n
        n_diff_states = self.n_diff_states
        conc_param = self.conc_param
        init_list = self.init_list  
        fix_list = self.fix_list
        var_list = self.var_list
        n_disturbance = self.n_disturbance    
        n_forecast = self.n_forecast 
        
        #change global flag var for smoothing alogorthm
        #self.flag == 0 is for EM algorithm
        self.flag = 1        
        
        #flag = 1 for parameter calibration
        flag=[1]
        
        #run bfgs algorithm for calibration of variance parameters
        #gtol is stopping criteria, eps is supposed to be step size
        res = minimize(self.kalman, init_list, flag, method='BFGS', options={'gtol': 1e-3,'eps': 1e-3,'disp': True, 'maxiter': 5000})            
        
        #run kalman filter for final time with optimized parameter values
        [LL, A, EA, P, EP, T, C, V, F, L, R, Q, K, H, sig_rec, KA, KP, KV, KF, KK, KL] = self.kalman(res.x,flag=2)
        
        self.flag = 2 

        if "cycle" in comp_dic:
            if "cycle" in var_list:
                damp = np.abs(res.x[2])*(1+(res.x[2]**2))**(-0.5)
                freq = (2.0*np.pi/(2.0 + np.exp(res.x[3])))
                period = 2.0*np.pi/freq
                print(damp)
                print(freq)
                print(period)
        
        #transform back parameters
        res.x = np.exp(res.x)
        
        #get concentrated parameter value
        sig_rec = float(sig_rec[0,0])
    
        #adjust for having concentrating values
        if conc_param != "none":
            F = [f*sig_rec for f in F]
            Q = Q*sig_rec
            H = H*sig_rec
            #yes both EP and P
            EP = [ep*sig_rec for ep in EP]
            P = [p*sig_rec for p in P]

        #these parameters have extra values that are needed for smoothing
        #but arent actually part of them
        #so create global var now
        self.A = A
        self.V = V
        self.F = F
        self.P = P
        self.EP = EP
        self.KA = KA
        self.KP = KP
        self.KV = KV
        self.KF = KF
        self.KK = KK
        self.KL = KL      

        #Create standardised prediction errors
        epsilon = []
        for t in range(0,n):
            epsilon.append(V[t]/np.sqrt(F[t]))
        
        kepsilon = []
        for t in range(0,n):
            kepsilon.append(KV[t]/np.sqrt(float(sig_rec)*KF[t]))
        
        #make adjustments for filter results
        KA = KA[1:-1]
        L3 = []
        for i in range(n_diff_states):
            L3.append([np.nan])
        L3 = np.mat(L3)
        KA.insert(0,L3)
        
        KP = KP[1:-1]
        kp_start = np.empty((n_diff_states,n_diff_states,))
        kp_start[:] = np.nan
        KP.insert(0,np.mat(kp_start))
        
        KF = KF[1:]
        KF.insert(0,np.mat([np.nan]))
        
        KV = KV[1:]
        KV.insert(0,np.mat([np.nan]))
        
        KP = [float(sig_rec)*kp for kp in KP]
        KF = [float(sig_rec)*kf for kf in KF]
        
        #make adjustments for filter results
        A = A[1:-1]
        L2 = []
        for i in range(n_diff_states):
            L2.append([np.nan])
        L2 = np.mat(L2)
        A.insert(0,L2)

        F = F[1:]
        F.insert(0,np.mat([np.nan]))
        
        V = V[1:]
        V.insert(0,np.mat([np.nan]))
        
        EP = EP[1:]
        ep_start = np.empty((n_diff_states,n_diff_states,))
        ep_start[:] = np.nan
        EP.insert(0,np.mat(ep_start))
        
        P = P[2:]
        p_start = np.empty((n_diff_states,n_diff_states,))
        p_start[:] = np.nan
        P.insert(0,np.mat(p_start))

        kstn_pred_err = []
        for matrix in kepsilon:
                kstn_pred_err.append(float(matrix[0]))
                
        stn_pred_err = []
        for matrix in epsilon:
                stn_pred_err.append(float(matrix[0]))
        
        if "cycle" in comp_dic and "cycle" in var_list:
            #get final param values from returned x0
            fin_v, fin_p = final_param_vals(comp_dic,conc_param,sig_rec,res,fix_list,damp,freq,period)
        else:
            fin_v, fin_p = final_param_vals(comp_dic,conc_param,sig_rec,res,fix_list)

        #create dataframe of results
        params = DataFrame(fin_p,columns=['component'])        
        params["variance"] = fin_v

        #create lists to populate dataframe
        count = 0
        if "level" in comp_dic:
            filter_level = []
            for matrix in A:
                filter_level.append(float(matrix[count]))
            count += 1
            #self.filter_state["level"] = filter_level
            self.fsd["level"] = filter_level
        if "slope" in comp_dic:
            filter_slope = []
            for matrix in A:
                filter_slope.append(float(matrix[count]))
            count += 1
            #self.filter_state["slope"] = filter_slope
            self.fsd["slope"] = filter_slope
            count += 1
        if "seasonal" in comp_dic:
            filter_seasonal = []
            for matrix in A:
                filter_seasonal.append(float(matrix[count]))
            count += 1
            #self.filter_state["seasonal"] = filter_seasonal
            self.fsd["seasonal"] = filter_seasonal
            count += 1
        if "irr" in comp_dic:
            if "level" in comp_dic and "seasonal" in comp_dic:
                #self.filter_state["irr"] = self.filter_state[self.filter_state.columns[0]] - self.filter_state["level"] - self.filter_state["seasonal"]
                if n_forecast > 0:
                    self.fsd["irr"] = self.fsd["y"] - self.fsd["level"][:-n_forecast] - self.fsd["seasonal"][:-n_forecast]
                else:
                    self.fsd["irr"] = self.fsd["y"] - self.fsd["level"] - self.fsd["seasonal"]
            elif "level" in comp_dic:
                if n_forecast > 0:
                    #self.filter_state["irr"] = self.filter_state[self.filter_state.columns[0]] - self.filter_state["level"]
                    self.fsd["irr"] = self.fsd["y"] - self.fsd["level"][:-n_forecast]
                else:
                    self.fsd["irr"] = self.fsd["y"] - self.fsd["level"]



        #self.filter_state["pred_err"] = [float(v) for v in V]
        self.fsd["pred_err"] = [float(v) for v in V]
        #self.filter_state["exp_level_var"] = [float(ep[0,0]) for ep in EP]
        self.fsd["exp_level_var"] = [float(ep[0,0]) for ep in EP]
        #self.filter_state["level_var"] = [float(p[0,0]) for p in P]
        self.fsd["level_var"] = [float(p[0,0]) for p in P]
        #self.filter_state["pred_err_var"] = [float(f) for f in F]
        self.fsd["pred_err_var"] = [float(f) for f in F]

        #change sign of LL back to + and change to float
        LL = float(-LL)
        nLL = LL/float(n - n_diff_states)
        
        #Akaike Information critereon
        AIC = (1/float(n))*(-2*n*LL + 2*(n_diff_states + n_disturbance))
        nAIC = (1/float(n))*(-2*n*nLL + 2*(n_diff_states + n_disturbance))
        

        #create goodness of fit dataframe
        fit = DataFrame(["log-likelihood", "nlog-likelihood","Aikake Information Criterion", "nAikake Information Criterion"],columns=['statistic'])
        fit["value"] = [LL,nLL,AIC, nAIC]
        
        #export global variables
        self.EA = EA
        self.T = T
        self.C = C
        self.params = params
        self.loglike = LL        
        self.stn_pred_err = stn_pred_err
        self.kstn_pred_err = kstn_pred_err
        self.AIC = AIC
        self.fit = fit
        self.L = L
        self.R = R
        self.Q = Q
        self.K = K
        self.H = H
        self.sig_rec = sig_rec
        
        #return filtered dataframe
        #return self.filter_state
        return self.fsd
        
            
    def smooth(self):    
        
        #import global variables
        KA = self.KA  
        KP = self.KP
        KV = self.KV
        KF = self.KF
        KK = self.KK
        KL = self.KL
        sig_rec = self.sig_rec
        V = self.V
        F = self.F
        L = self.L
        K = self.K
        H = self.H
        Q = self.Q
        R = self.R
        A = self.A
        P = self.P
        C = self.C
        n = self.n
        Z = self.Z
        T = self.T    
        EA = self.EA
        EP = self.EP
        comp_dic = self.comp_dic
        n_diff_states = self.n_diff_states
        flag = self.flag
        n_forecast = 0

        if flag == 2:
            n_forecast = self.n_forecast
            n = n + n_forecast

        #what do we need here and what do we not?
        #KF = [float(sig_rec)*kf for kf in KF]
        #KP = [float(sig_rec)*kp for kp in KP]
        KP = KP[1:]
        

        #Classical Fixed Interval Smoothing Algorithm
        #create list of smoothed level
        #may need to seed with what we deleted from A before
        S = [A[-1]]
        #S = [fin_a]
        for t in range(n,-1,-1):
            S.append(A[t-1] + P[t-1] * T.T * np.linalg.pinv(EP[t-1]) * (S[n - t] - T*A[t-1] - C) )
        cfis = S
        cfis = cfis[:-2]

        #Fixed Interval Smoothing Algorithm
        #r2 is the smoothing state cumulant
        r1 = [np.mat([0]*n_diff_states).T]
        for t in range(n-1,-1,-1):
            r1.append(  Z.T*np.linalg.pinv(F[t])*V[t] + L[t].T*r1[n-1-t] ) 
        r2 = []
        for e in reversed(r1):
            r2.append(e)
        fis = []
        for t in range(0,n):
            fis.append(EA[t] + EP[t]*r2[t])   
        

        #Koopman FIS    
        #kr2 is the smoothing state cumulant
        kr1 = [np.mat([0]*n_diff_states).T]
        for t in range(n-1,-1,-1):
            kr1.append(  Z.T*np.linalg.pinv(KF[t])*KV[t] + KL[t].T*kr1[n-1-t] ) 
        kr2 = []
        for e in reversed(kr1):
            kr2.append(e)
        kfis = []
        for t in range(0,n):
            kfis.append(KA[t] + KP[t]*kr2[t])                            


        #Fixed Interval Smoothed State Variance
        #N is the smoothing state variance cumulant 
        tF = F[1:]
        tF.insert(0,np.mat([np.nan]))                           
        n1 = [np.zeros((n_diff_states,n_diff_states))]
        for t in range(n-1,-1,-1):
            n1.append(  Z.T*np.linalg.pinv(tF[t])*Z + L[t].T*n1[n-1-t]*L[t] ) 
        N = []
        for e in reversed(n1):
        #for e in n1:
            N.append(e)

        #KN is the smoothing state variance cumulant 
        ktF = KF[1:]
        ktF.insert(0,np.mat([np.nan]))                           
        kn1 = [np.zeros((n_diff_states,n_diff_states))]
        for t in range(n-1,-1,-1):
            kn1.append(  Z.T*np.linalg.pinv(ktF[t])*Z + KL[t].T*kn1[n-1-t]*KL[t] ) 
        KN = []
        for e in reversed(kn1):
        #for e in n1:
            KN.append(e)





        V2 = []
        for t in range(0,n):
            #we want N(t-1)
            V2.append(EP[t] - EP[t]*N[t]*EP[t])
        self.VV = [float(vv[0,0]) for vv in V2]


        KV2 = []
        for t in range(0,n):
            #we want N(t-1)
            KV2.append(KP[t] - KP[t]*KN[t]*KP[t])
        self.KVV = [float(kvv[0,0]) for kvv in KV2]





        #Smoothed Distrubances
        #smoothed observation disturbances
        eps = []
        for t in range(0,n):
            u = np.linalg.pinv(F[t])*V[t] - K[t].T*r2[t+1]
            eps.append(  H*u  )
        #smoothed state disturbances
        eta = []
        for t in range(0,n):
            eta.append(  Q*R.T*r2[t+1]  )

        keps = []
        for t in range(0,n):
            ku = np.linalg.pinv(KF[t])*KV[t] - KK[t].T*kr2[t+1]
            keps.append(  H*ku  )
        #smoothed state disturbances
        keta = []
        for t in range(0,n):
            keta.append(  Q*R.T*kr2[t+1]  )




        #Smoothed Distrubances Variances
        #smoothed observation disturbance variance
        var_eps = []
        for t in range(0,n):
            D = np.linalg.pinv(F[t]) + K[t].T*N[t+1]*K[t]
            var_eps.append(H - H*D*H)
        #smoothed state disturbances variance
        var_eta = []
        for t in range(0,n):
            var_eta.append(Q - Q*R.T*N[t]*R*Q)
        
        
        kvar_eps = []
        for t in range(0,n):
            KD = np.linalg.pinv(KF[t]) + KK[t].T*KN[t+1]*KK[t]
            kvar_eps.append(H - H*KD*H)
        #smoothed state disturbances variance
        kvar_eta = []
        for t in range(0,n):
            kvar_eta.append(Q - Q*R.T*KN[t]*R*Q)
        

        
        
        #Fast Smoothing Algorithm
        fs = [EA[0] + EP[0]*r2[0]]
        for t in range(0,n):
            fs.append(T*fs[t] + R*eta[t])
        fs = fs[:-1]

        kfs = [KA[0] + KP[0]*kr2[0]]
        for t in range(0,n):
            kfs.append(T*kfs[t] + R*keta[t])
        kfs = kfs[:-1]
        
        # g = []
        # for i in kfs:
        #     g.append(i[0,0])
        # self.kfs = g
        

        #self.fs_m0 = [float(m[0]) for m in fs]
        #self.fs_m1 = [float(m[1]) for m in fs]
        #self.fs_m2 = [float(m[2]) for m in fs]
        
        #import pylab
        #import matplotlib.pyplot as plt
        #plt.plot(self.fs_m1)
        #pylab.show()
#        
#        if self.c_count >= 20 and self.c_count < 25:
#            
#
#        elif self.c_count == 26:
#            assert(False)
#        self.c_count += 1


        


        #Create Dataframe of results
        count = 0
        if "level" in comp_dic:
            cfis_level = [float(m[count]) for m in reversed(cfis)]
            fis_level = [float(m[count]) for m in fis]
            fs_level = [float(m[count]) for m in fs]
            
            s_obs_dis_level = [float(m[count]) for m in eps]
            s_obs_dis_var_level = [float(m[count,count]) for m in reversed(var_eps)]
            
            #calc 95% confidence intervals
            up = []
            bottom = []
            for index, mu in enumerate(s_obs_dis_var_level):
                u = fs_level[index] + 1.64*np.sqrt(mu)
                up.append(u)
                b = fs_level[index] - 1.64*np.sqrt(mu)
                bottom.append(b)

            #self.smooth_state2["up"] = up
            self.ss2d["up"] = up
            #self.smooth_state2["bottom"] = bottom
            self.ss2d["bottom"] = bottom


            s_state_dis_level = [float(m[count]) for m in eta]
            s_state_dis_var_level = [float(m[count,count]) for m in reversed(var_eta)]

            #self.smooth_state2["cfis_level"] = cfis_level
            self.ss2d["cfis_level"] = cfis_level
            #self.smooth_state2["fis_level"] = fis_level
            self.ss2d["fis_level"] = fis_level
            #self.smooth_state2["fs_level"] = fs_level
            self.ss2d["fs_level"] = fs_level
            
            #self.smooth_state2["s_obs_dis_level"] = s_obs_dis_level
            self.ss2d["s_obs_dis_level"] = s_obs_dis_level
            #self.smooth_state2["s_obs_dis_var_level"] = s_obs_dis_var_level
            self.ss2d["s_obs_dis_var_level"] = s_obs_dis_var_level
            
            #self.smooth_state2["s_state_dis_level"] = s_state_dis_level
            self.ss2d["s_state_dis_level"] = s_state_dis_level
            #self.smooth_state2["s_state_dis_var_level"] = s_state_dis_var_level
            self.ss2d["s_state_dis_var_level"] = s_state_dis_var_level
            
            smooth_level_var = [float(s[count,count]) for s in V2]
            #self.smooth_state2["smooth_level_var"] = smooth_level_var
            self.ss2d["smooth_level_var"] = smooth_level_var
            
            smooth_level_var_cum = [float(s[count,count]) for s in N]
            #self.smooth_state2["smooth_level_var_cum"] = smooth_level_var_cum[1:]
            self.ss2d["smooth_level_var_cum"] = smooth_level_var_cum[1:]
            
            smooth_level_cum = [float(s[count,count]) for s in r2]
            #self.smooth_state2["smooth_level_cum"] = smooth_level_cum[1:]
            self.ss2d["smooth_level_cum"] = smooth_level_cum[1:]
            


            
            
            count += 1
        if "slope" in comp_dic:
            cfis_slope = [float(m[count]) for m in reversed(cfis)]
            fis_slope = [float(m[count]) for m in fis]
            fs_slope = [float(m[count]) for m in fs]
            
            #s_obs_dis_slope = [float(m[count-1]) for m in eps]
            #s_obs_dis_var_slope = [float(m[count,count]) for m in reversed(var_eps)]
            
            #s_state_dis_slope = [float(m[count]) for m in eta]
            #s_state_dis_var_slope = [float(m[count,count]) for m in reversed(var_eta)]
            
            #self.smooth_state2["cfis_slope"] = cfis_slope
            self.ss2d["cfis_slope"] = cfis_slope
            #self.smooth_state2["fis_slope"] = fis_slope
            self.ss2d["fis_slope"] = fis_slope
            #self.smooth_state2["fs_slope"] = fs_slope  
            self.ss2d["fs_slope"] = fs_slope  
            
            #self.smooth_state2["s_obs_dis_slope"] = s_obs_dis_slope
            #self.smooth_state2["s_obs_dis_var_slope"] = s_obs_dis_var_slope
            
            #self.smooth_state2["s_state_dis_slope"] = s_state_dis_slope
            #self.smooth_state2["s_state_dis_var_slope"] = s_state_dis_var_slope
            
            smooth_slope_var = [float(s[count,count]) for s in V2]
            #self.smooth_state2["smooth_slope_var"] = smooth_slope_var
            self.ss2d["smooth_slope_var"] = smooth_slope_var
            count += 1
        
        if "seasonal" in comp_dic:
            cfis_seasonal = [float(m[count]) for m in reversed(cfis)]
            fis_seasonal = [float(m[count]) for m in fis]
            fs_seasonal = [float(m[count]) for m in fs]
            
            #s_obs_dis_slope = [float(m[count-1]) for m in eps]
            #s_obs_dis_var_slope = [float(m[count,count]) for m in reversed(var_eps)]
            
            #s_state_dis_slope = [float(m[count]) for m in eta]
            #s_state_dis_var_slope = [float(m[count,count]) for m in reversed(var_eta)]
            
            #self.smooth_state2["cfis_seasonal"] = cfis_seasonal
            self.ss2d["cfis_seasonal"] = cfis_seasonal
            #self.smooth_state2["fis_seasonal"] = fis_seasonal
            self.ss2d["fis_seasonal"] = fis_seasonal
            #self.smooth_state2["fs_seasonal"] = fs_seasonal  
            self.ss2d["fs_seasonal"] = fs_seasonal  
            
            #self.smooth_state2["s_obs_dis_slope"] = s_obs_dis_slope
            #self.smooth_state2["s_obs_dis_var_slope"] = s_obs_dis_var_slope
            
            #self.smooth_state2["s_state_dis_slope"] = s_state_dis_slope
            #self.smooth_state2["s_state_dis_var_slope"] = s_state_dis_var_slope
            
            smooth_seasonal_var = [float(s[count,count]) for s in V2]
            #self.smooth_state2["smooth_seasonal_var"] = smooth_seasonal_var
            self.ss2d["smooth_seasonal_var"] = smooth_seasonal_var
            count += 1

        #Add Resids
        if "level" in comp_dic and "seasonal" in comp_dic:
            #self.smooth_state2["irr"] = self.smooth_state2[self.smooth_state2.columns[0]] - self.smooth_state2["fis_level"] - self.smooth_state2["fis_seasonal"]
            if n_forecast > 0:
                self.ss2d["irr"] = self.ss2d["y"] - self.ss2d["fis_level"][:-n_forecast] - self.ss2d["fis_seasonal"][:-n_forecast]
            else:
                self.ss2d["irr"] = self.ss2d["y"] - self.ss2d["fis_level"] - self.ss2d["fis_seasonal"]
        elif "level" in comp_dic:
            #self.smooth_state2["irr"] = self.smooth_state2[self.smooth_state2.columns[0]] - self.smooth_state2["fis_level"]
            if n_forecast > 0:
                self.ss2d["irr"] = self.ss2d["y"] - self.ss2d["fis_level"][:-n_forecast]
            else:
                self.ss2d["irr"] = self.ss2d["y"] - self.ss2d["fis_level"]

        
        if flag == 0:
            B = 0
            c = 0
            for t in range(0,n):
                


                c += eps[t][0]**2 + var_eps[t]
                
                if t >= n_diff_states:
                    B += eta[t-1]*eta[t-1].T + var_eta[t-1]    
            
            xsi_irr = float(c/float(n))

            r_x0 = [xsi_irr]
            count = 0
            if "level" in comp_dic:
                xsi_level = B[count,count]/float(n-1)
                r_x0.append(xsi_level)
                count += 1
            if "slope" in comp_dic:
                xsi_slope = B[count,count]/float(n-1)
                r_x0.append(xsi_slope)
                count += 1
            if "cycle" in comp_dic:
                xsi_cycle = B[count,count]/float(n-1)
                print(B[count,count])
                r_x0.append(xsi_cycle)
                count += 1
            if "seasonal" in comp_dic:
                xsi_seasonal = B[count,count]/float(n-1)
                r_x0.append(xsi_seasonal)
                count += 1

            


            return r_x0
        else:
            #return self.smooth_state2
            return self.ss2d
      
        
    def diagnostics(self,k):
            
        #irr = self.smooth_state2["irr"]
            
        #import global variables
        stn_pred_err = self.kstn_pred_err
        stn_pred_err2 = self.kstn_pred_err
        
        #n_diff_states = self.n_diff_states
        n_diff_states = self.n_diff_states
        
        n_disturbance = self.n_disturbance
        n = self.n
        
        #create new n var with n_diff_states removed
        n_minus_comp = n - n_diff_states
        
        
        #change to array
        stn_pred_err = np.asarray(stn_pred_err)
        
        #--------------------------------------------------------------#
        #Independence Test
        #--------------------------------------------------------------#
        #create autocorrelations for all k lags

        rk = []
        for i in range(1,k+1):
            num_end = n - i  
            #calculate numerator and denominator
            num = 0
            den = 0
            for t in range(0,n):
                if t+1 > n_diff_states:
                    #denominator
                    den += (stn_pred_err[t] - np.mean(stn_pred_err[n_diff_states:]))**2
                    #numerator
                    if t+1 <= num_end:
                        num += (stn_pred_err[t] - np.mean(stn_pred_err[n_diff_states:]))*(stn_pred_err[t+i] - np.mean(stn_pred_err[n_diff_states:]))
            #calculate autocorrelations
            rk.append(num/den)
        #calculate Box-Ljung statistic

        sacf = []
        #Y = stn_pred_err2
        Y = stn_pred_err2[n_diff_states:]
        for e in range(1,11):
            flen = float(len(Y))
            ybar = float(sum([y for y in Y])) / flen
            D = sum([(y-ybar)**2 for y in Y])
            N = sum([ (y-ybar)* (ytpk -ybar) for (y, ytpk) in zip(Y[:-e],Y[e:])])
            sacf.append(N/D)
        
        self.sacf = sacf
        
        Qk = 0
        for j in range(0,len(rk)):
            tau = j + 1
            Qk += (rk[j]**2.0)/(float(n_minus_comp-tau))
        Qk = n_minus_comp*(n_minus_comp+2.0)*Qk      
        
        #calculate critical value for BL test of independence
        Qc = sp.stats.chi2.ppf(0.95, k - n_disturbance + 1)
        
        #assumption satisfied?
        if Qk > Qc:
            Qx = "X"
        else:
            Qx = "O"
        #--------------------------------------------------------------#
        #Homoskedasticity Test
        #--------------------------------------------------------------#
        
        #break up into thirds. h is number of obs in each third
        h = np.round((n_minus_comp)/3.0)
        
        #calculate heteroskedasticy statistic
        den = 0
        num = 0
        for t in range(0,n):
            #den
            if t + 1 > n_diff_states and t + 1 <= n_diff_states + h:
                den += stn_pred_err[t]**2
            #num
            if t + 1 > n - h:
                num += stn_pred_err[t]**2
        #calculate heteroskedatsicy stat
        Hh = num/den
        #if smaller than 1, use inverse
        #page 55 koopman
        if Hh < 1:
            Hh = float(1/Hh)
        
        #calculate critical value for BL test of independence
        Hc = sp.stats.f.ppf(0.975, h,h)
        
        #assumption satisfied?
        if Hh > Hc:
            Hx = "X"
        else:
            Hx = "O"
        #--------------------------------------------------------------#
        #Normality Test
        #--------------------------------------------------------------#

        #skew
        num = 0
        den = 0
        for t in range(0,n):
            if t + 1 > n_diff_states:
                num += (stn_pred_err[t] - np.mean(stn_pred_err[n_diff_states:]))**3            
                den += (stn_pred_err[t] - np.mean(stn_pred_err[n_diff_states:]))**2         
        num = num/n_minus_comp
        den = den/n_minus_comp
        den = den**(3.0/2.0)
        skew = num/den

        #kurtosis
        num = 0
        den = 0
        for t in range(0,n):
            if t + 1 > n_diff_states:
                num += (stn_pred_err[t] - np.mean(stn_pred_err[n_diff_states:]))**4
                den += (stn_pred_err[t] - np.mean(stn_pred_err[n_diff_states:]))**2
        num = num/n_minus_comp
        den = den/n_minus_comp
        den = den**2  
        kurt = num/den

        #Jarque-Bera Normality test statistic
        N = n_minus_comp*( ((skew**2)/6.0) + (((kurt - 3)**2)/24.0) )
        
        #critical value 
        Nc = sp.stats.chi2.ppf(0.95, 2)
        
        #assumption satisfied?
        if N > Nc:
            Nx = "X"
        else:
            Nx = "O"
        
        #--------------------------------------------------------------#
        
        #create test result dataframe
        tests = DataFrame(["independence", "homoskedasticity", "normality"],columns=[''])
        tests["statistic"] = ["Q("+str(k)+")","H("+str(int(h))+")","N"]
        tests["value"] = [Qk,Hh,N]
        tests["95% critical value"] = [Qc,Hc,Nc]
        tests["assumption satisfied"] = [Qx,Hx,Nx]

        #export global variables
        self.Qk = Qk
        self.autocorr = rk
        self.tests = tests

        #return dataframe
        return tests
  
   