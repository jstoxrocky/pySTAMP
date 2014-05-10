import numpy as np 
       
def concat(Qnw, Qse):

    n = len(Qnw)
    m = len(Qse)

    Qsw = np.zeros((m,n))
    Qne = np.zeros((n,m))

    Ql = np.concatenate((Qnw, Qsw), axis=0)
    Qr = np.concatenate((Qne, Qse), axis=0)
    Q = np.concatenate((Ql, Qr), axis=1)

    return Q

def set_param_vals(comp_dic,conc_param,var_list,flag,x0,n_diff_states,y,T,s):
            
        
            #set parameters values
            count = 0
            if "irr" in comp_dic and conc_param != "irr":
                if "irr" in var_list:
                    if flag == 0:
                        q_irr = x0[count]
                    else:
                        q_irr = np.exp(x0[count])
                    count += 1 
                else:
                    q_irr = 0
                #count += 1   
                    
            if "level" in comp_dic and conc_param != "level":
                if "level" in var_list:
                    if flag == 0:
                        q_level = x0[count]
                    else:
                        q_level = np.exp(x0[count])
                    count += 1 
                else:
                    q_level = 0
                #count += 1
                    
            if "slope" in comp_dic and conc_param != "slope":

                if "slope" in var_list:
                    if flag == 0:
                        q_slope = x0[count]
                    else:
                        q_slope = np.exp(x0[count])
                    count += 1   
                else:
                    q_slope = 0 
                #count += 1
            
            if "cycle" in comp_dic and conc_param != "cycle":
                if "cycle" in var_list:
                    q_cycle = np.exp(x0[count])
                    damp = 0.9
                    freq = 2.0*np.pi/14.0
                else:
                    q_cycle = 0.0
                    damp = 0.9
                    freq = 2.0*np.pi/14.0
            
            if "seasonal" in comp_dic and conc_param != "seasonal":

                if "seasonal" in var_list:
                    if flag == 0:
                        q_seasonal = x0[count]
                    else:
                        q_seasonal = np.exp(x0[count])
                    count += 1   
                else:
                    q_seasonal = 0 
                #count += 1                

            #    if "cycle" in var_list:
            #        if flag == 0:
            #            q_cycle = np.exp(x0[count])
            #            damp = (np.abs(x0[count+1])*(1+(x0[count+1]**2))**(-0.5))  
            #            freq = (2.0*np.pi/(2.0 + np.exp(x0[count+2])))
            #            #freq = (2.0*np.pi/(2.0 + np.exp(x0[count+2]))) + (0.418879020479/(1 + np.exp(-x0[count+2])))
            #        elif flag == -1:
            #            q_cycle = np.exp(x0[count])
            #            damp = 0.9 
            #            freq = 2.0*np.pi/14.0
            #        else:
            #            q_cycle = np.exp(x0[count])
            #            damp = (np.abs(x0[count+1])*(1+(x0[count+1]**2))**(-0.5))  
            #            #freq = (2.0*np.pi/(2.0 + np.exp(x0[count+2]))) + (0.418879020479/(1 + np.exp(-x0[count+2])))
            #            freq = (2.0*np.pi/(2.0 + np.exp(x0[count+2])))
            #        count += 1 
            #    
            #    else:
            #        q_cycle = 0.0 
            #        damp = 1.0
            #        freq = 2.0*np.pi/14.0
            #        
            #    #count += 1
            #elif "cycle" in comp_dic and conc_param == "cycle": #in case cycle is concentrated out we still need damp & freq
            #    damp = (np.abs(x0[count+1])*(1+(x0[count+1]**2))**(-0.5))  
            #    freq = (2.0*np.pi/(2.0 + np.exp(x0[count+2])))
            #    #freq = (2.0*np.pi/(2.0 + np.exp(x0[count+2]))) + (0.418879020479/(1 + np.exp(-x0[count+2])))


            #try:
            #    print(damp)
            #    print(2.0*np.pi/freq)
            #except:
            #    pass

            #create disturbance matrices H and Q
            #restricted to unity if concentrated out
            if "irr" in comp_dic:
                if conc_param == "irr":
                    q_irr = 1
                H = q_irr
        
        
            #Q = np.matrix(np.identity(n_diff_states))
            Q = np.zeros((n_diff_states,n_diff_states))

            count = 0
            if "level" in comp_dic:
                if conc_param == "level":
                    q_level = 1
                Q[count,count] = q_level
                count += 1
            if "slope" in comp_dic:
                if conc_param == "slope":
                    q_slope = 1
                Q[count,count] = q_slope
                count += 1
            if "cycle" in comp_dic:
                if conc_param == "cycle":
                    q_cycle = 1
                
                #upper left
                Q[count,count] = q_cycle*(1.0 - damp**2)
                #Q[count,count] = q_cycle
                T[count,count] = damp*np.cos(freq)
                #upper right
                T[count,count+1] = damp*np.sin(freq)
                count += 1
                #bottom left
                T[count,count-1] = -damp*np.sin(freq)
                #bottom right
                Q[count,count] = q_cycle*(1.0 - damp**2)
                #Q[count,count] = q_cycle
                T[count,count] = damp*np.cos(freq)
                count += 2

            if "seasonal" in comp_dic:
                if conc_param == "seasonal":
                    q_seasonal = 1
                Q[count,count] = q_seasonal

                count += s - 1


            #Initialize state and state variance matrices
            a0 = 0
            a0 = [a0 for i in range(n_diff_states)]
            #a0[0] = y[0]
            #a0[1] = y[0]
            a0 = np.matrix(a0)
            a0 = a0.T
            A = [a0]
                
            p0 = 10000
            p0 = np.matrix(np.identity(n_diff_states))*p0
            P = [p0]

            return(H,Q,A,P,T)
            
            
def k_filter(n,T,A,C,P,R,Q,y,Z,D,H,n_diff_states,n_disturbance,n_forecast=0):

            
            #LL_term2_rec: loglikelihood recursion term 2
            #LL_term3_rec: loglikelihood recursion term 3
            #sig_rec = 0: concentrated likelihood recursion parameter 
            
            LL_term2_rec = 0
            LL_term3_rec = 0
            sig_rec = 0
            
            #***********************************************#
            #Harvey: Time Series Model (TSM): Kalman Filter
            #***********************************************#
            
            #EA: Expected state level (equal to filtered state in SSM)
            #EP: Expected state variance (equal to filtered state in SSM)
            #A: Filtered state (not calculated in SSM)
            #P: Filtered state variance (not calculated in SSM)
            #V: One step ahead forecast error (innovations)
            #F: One step ahead forecast error variance
            #K: Kalman gain 1
            #L: Kalman gain 2

            EA = []
            EP = []
            V = []
            F = []
            K = []
            L = []

            #*********************************************************************************#
            #Durbin & Koopman: Time Series Analysis by State Space Method (SSM): Kalman Filter
            #*********************************************************************************#
        
            #KA: Filtered state (equal to expected state in TSM)
            #KP: Filtered state variance (equal to expected state in TSM)
            #KV: One step ahead forecast error (innovations)
            #KF: One step ahead forecast error variance
            #KK: Kalman gain 1
            #KL: Kalman gain 2
            #KM: Kalman gain 3
            
            KV = []
            KF = []
            KL = []
            KK = []
            KM = []

            #***********************************************#
            #Durbin & Koopman: (SSM): Exact Initial Filter (EI)
            #***********************************************#
            
            #xA: EI Filtered state
            #xPinf: EI Diffuse part of filtered state variance
            #xPstar: EI Non-diffuse part of filtered state variance
            #xV: EI One step ahead forecast error (innovations)
            #xFinf: EI Diffuse part of one step ahead forecast error variance
            #xFstar: EI Non-diffuse part of one step ahead forecast error variance
            #xF1: Building block 1 for EI forecast error variance
            #xF2: Building block 2 for EI forecast error variance
            #xK0: Kalman gain 1.0
            #xK1: Kalman gain 1.1
            #xL0: Kalman gain 2.0
            #xL1: Kalman gain 2.1
            #xMinf: EI Diffuse part of Kalman gain 3   
            #xMstar: EI Non-diffuse part of Kalman gain 3             
            #xflag: Flag allowing EI recursions
            #f_is_singular: Flag controlling collapse to regular D&K recursions
            
            xA = A[:]
            xPinf = [np.matrix(np.identity(n_diff_states))]
            xPstar = [np.zeros((n_diff_states,n_diff_states))]
            xV = []
            xFinf = []
            xFstar = []
            xF1 = []
            xF2 = []
            xK0 = []
            xK1 = []           
            xL1 = []
            xL0 = []
            xMinf = []
            xMstar = []
            xflag = 0
            f_is_singular = False

            #Begin Kalman Filter Recurisions
            for t in range(0,n + n_forecast):          
                
                if xflag == 0:
                    
                    #Exact Initial Kalman Filter
                    xMinf.append(xPinf[t]*Z.T)
                    xFinf.append(Z*xPinf[t]*Z.T)

                    xF1.append(np.linalg.pinv(xFinf[t]))
                    xFstar.append(Z*xPstar[t]*Z.T + H)
                
                    xF2.append(-np.linalg.pinv(xFinf[t])*xFstar[t]*np.linalg.pinv(xFinf[t]))
                    xMstar.append(xPstar[t]*Z.T)
                    
                    xK0.append(T*xMinf[t]*xF1[t])
                    xL0.append(T - xK0[t]*Z)
                    
                    xK1.append(T*xMstar[t]*xF1[t] + T*xMinf[t]*xF2[t])
                    xL1.append(-xK1[t]*Z)
                    
                    xV.append(y[t] - Z*xA[t])
                    xA.append(T*xA[t] + xK0[t]*xV[t])
                    
                    xPinf.append(T*xPinf[t]*xL0[t].T)
                    xPstar.append(T*xPinf[t]*xL1[t].T + T*xPstar[t]*xL0[t].T + R*Q*R.T)

                    #print(xPinf[t+1])
                    #if xPinf matrix is all zeros
                    #the EI Kalman filter can be collapsed
                    if np.allclose(xPinf[t+1], np.zeros((n_diff_states,n_diff_states))):
                        xflag = 1
                        KA = xA
                        KV = xV
                        KP = xPstar
                        KF = xFstar
                        KK = xK1
                        KL = xL1

                if f_is_singular:
                                 
                    #SSM Kalman Recursions
                    try:
                        KV.append(y[t] - Z*KA[t])
                    except:
                        KV.append(np.mat([0.0]))
                    KF.append(Z*KP[t]*Z.T + H)
                                                
                    KM.append(KP[t]*Z.T)
                    try:
                        KK.append(T*KP[t]*Z.T*np.linalg.pinv(KF[t]))
                    except:
                        KK.append(np.mat([0.0]))
                    KL.append(T - KK[t]*Z)
                                                    
                    KA.append(T*KA[t] + KK[t]*KV[t])               
                    KP.append(T*KP[t]*KL[t].T + R*Q*R.T)
                                                    

                    #KV.append(np.mat([0.0]))
                    #KK.append(np.mat([0.0]))

                    #KV = y[t] - Z*KA[t]
                    #KK = T*KP[t]*Z.T*np.linalg.pinv(KF[t])




                    #EA.append(T*A[t] + C)
                    # EP.append(T*P[t]*T.T + R*Q*R.T)

                    # A.append(EA[t])
                    #V=0


                    # P.append(EP[t])
                    #EP[t] - EP[t]*Z.T*np.linalg.pinv(F[t])*Z*EP[t]
                    
                    #K = T*EP[t]*Z.T*np.linalg.pinv(F[t])
                    #L = T - K[t]*Z 

                    #if K = 0
                    #L = T


                    #L = T - T*EP[t]*Z.T*np.linalg.pinv(F[t])*Z 

                    #P = EP - EP * Z.T * np.linalg.pinv(F[t]) * Z * EP



                    #EP = T*EP*T.T + R*Q*R.T

                    #T*KP[t]*KL[t].T + R*Q*R.T


                if xflag == 1:
                    f_is_singular = True
                
                #TSM Kalman Recursions
                #try except is used for forecasting
                
                EA.append(T*A[t] + C)
                EP.append(T*P[t]*T.T + R*Q*R.T)

                try:       
                    V.append(y[t] - Z*EA[t] - D)
                except:
                    V.append(np.mat([0.0]))

                F.append(Z*EP[t]*Z.T + H)
                   
                K.append(  T*EP[t]*Z.T*np.linalg.pinv(F[t])  )          
                L.append(  T - K[t]*Z  )  

                A.append(EA[t] + EP[t]*Z.T*np.linalg.pinv(F[t])*V[t])
                P.append(EP[t] - EP[t]*Z.T*np.linalg.pinv(F[t])*Z*EP[t])



                    
                #A.append(EA[t])
                #P.append(EP[t])



                # #Likelihood Function
                # if t >= n_diff_states + 1 - 1:
                
                #     #calculate concentrated parameter recursions
                #     sig_curr = V[t].T*np.linalg.pinv(F[t])*V[t]
                #     sig_rec = sig_rec + sig_curr
                    
                #     #calculate loglikelihood
                #     LL_term2_curr = np.log(np.linalg.det(F[t]))
                #     LL_term2_rec = LL_term2_rec + LL_term2_curr
                    
                #     LL_term3_curr = V[t].T*np.linalg.pinv(F[t])*V[t]
                #     LL_term3_rec = LL_term3_rec + LL_term3_curr
                # else:
                #     LL_term2_rec = 0
                #     LL_term3_rec = 0
            
                #Likelihood Function
                if t >= n_diff_states + 1 - 1:
                
                    #calculate concentrated parameter recursions
                    sig_curr = KV[t].T*np.linalg.pinv(KF[t])*KV[t]
                    sig_rec = sig_rec + sig_curr
                    
                    #calculate loglikelihood
                    LL_term2_curr = np.log(np.linalg.det(KF[t]))
                    LL_term2_rec = LL_term2_rec + LL_term2_curr
                    
                    LL_term3_curr = KV[t].T*np.linalg.pinv(KF[t])*KV[t]
                    LL_term3_rec = LL_term3_rec + LL_term3_curr
                else:
                    LL_term2_rec = 0
                    LL_term3_rec = 0

            #used for EM algorithm initialisation
            dVar = (1/float(n-n_diff_states))*LL_term3_rec
            
            #Calculate concentrated parameter
            sig_rec = (1/float(n - n_diff_states))*sig_rec
            sig1 = -(n/2.0)*np.log(2*np.pi)
            sig2 = -((n-float(n_diff_states))/2.0)
            sig3 = -((n-float(n_diff_states))/2.0)*np.log(sig_rec)
            sig4 = -0.5*LL_term2_rec
            

            #Calculate concetrated loglikelihood
            LLc = sig1 + sig2 + sig3 + sig4
            
            #Calculate loglikelihood (as if there were no concentration)
            #LL is the term we want to maximise
            #LL is what we want to know about
            #-LL is only used to change the min function into a max function
            LL_term1 = -(n/2.0)*np.log(2*np.pi)
            LL = LL_term1 - 0.5*LL_term2_rec - 0.5*LL_term3_rec 
            LL = LL/float(n)
            
            try:
            
                return LL, LLc, dVar, V, F, L, EA, EP, K, H, Q, R, A, P, C, sig_rec, KA, KP, KV, KF, KK, KL
            
            except:
                
                KA = EA
                KP = EP
                KV = V 
                KF = F
                KK = K
                KL = L
                
                return LL, LLc, dVar, V, F, L, EA, EP, K, H, Q, R, A, P, C, sig_rec, KA, KP, KV, KF, KK, KL

def final_param_vals(comp_dic,conc_param,sig_rec,res,fix_list,damp=None,freq=None,period=None):
        
        #create param values 
        p = []
        v = []
        count =  0
        if "irr" in comp_dic:
            p.append("irr")
            if conc_param == "irr":
                x_irr = float(sig_rec)
            elif conc_param in comp_dic:
                x_irr = float(sig_rec*res.x[count])
                count += 1
            else:
                x_irr = float(res.x[count])
                count += 1
            if "irr" in fix_list:
                v.append(0)
            else:
                v.append(x_irr)
            
        if "level" in comp_dic:
            p.append("level")
            if conc_param == "level":
                x_level = float(sig_rec)
            elif conc_param in comp_dic:
                x_level = float(sig_rec*res.x[count])
                count += 1
            else:
                x_level = float(res.x[count])
                count += 1
            if "level" in fix_list:
                v.append(0)
            else:
                v.append(x_level)
                
        if "slope" in comp_dic:
            p.append("slope")
            if conc_param == "slope":
                x_slope = float(sig_rec)
            elif conc_param in comp_dic:
                x_slope = float(sig_rec*res.x[count])
                count += 1
            else:
                x_slope = float(res.x[count])
                count += 1
            if "slope" in fix_list:
                v.append(0)
            else:
                v.append(x_slope)
        if "cycle" in comp_dic:
            p.append("cycle")
            
            if conc_param == "cycle":
                x_cycle = float(sig_rec)
            elif conc_param in comp_dic:
                x_cycle = float(sig_rec*res.x[count])
                count += 1
            else:
                x_cycle = float(res.x[count])
                count += 1
            if "cycle" in fix_list:
                v.append(0)
            else:
                p.append("damping factor")
                p.append("frequency")
                p.append("period")
                v.append(x_cycle)
                v.append(damp)
                count += 1
                v.append(freq)
                count += 1
                v.append(period)
                count += 1
        if "seasonal" in comp_dic:
            p.append("seasonal")
            if conc_param == "seasonal":
                x_seasonal = float(sig_rec)
            elif conc_param in comp_dic:
                x_seasonal = float(sig_rec*res.x[count])
                count += 1
            else:
                x_seasonal = float(res.x[count])
                count += 1
            if "seasonal" in fix_list:
                v.append(0)
            else:
                v.append(x_seasonal)


        return v, p
        
def spectral_density(y):
        
        y_bar = np.mean(y)
        T = len(y)
        c = []
        
        c0 = 0
        for t in range(0,T):
            c0 += (y[t] - y_bar)**2
        c0 = c0/float(T)
        
        for tau in range(1,T-1):
            curr_c = 0
            for t in range(tau+1,T):
                curr_c += (y[t-1] - y_bar)*(y[t-tau-1] - y_bar)
            c.append(curr_c)   
                    
        ssd = 0
        
        ssd = []
        term1 = 1/(2*np.pi)
        term2 = c0
        
        #generate numbers from 0 to pi
        lam = []
        x = 0
        
        while x <= np.pi:
            lam.append(x)
            x += 0.015
        
        #create ssd
        for x in lam:
            
            term3 = 0
            for tau in range(0,T-2):
                term3 += c[tau]*np.cos(x*(tau+1))
            term3 = term3*2
            
            ssd.append( term1*(term2 + term3) )
        
        
        def triangle(rows):

            for rownum in range(rows):
                newValue=1
                PrintingList = [newValue]
                for iteration in range (rownum):
                    newValue = newValue * ( rownum-iteration ) * 1 / ( iteration + 1 )
                    PrintingList.append(int(newValue))
            while len(PrintingList) > rows/2.0 + 1:
                PrintingList = PrintingList[1:]
            return PrintingList
        
        
        m = len(ssd)
        c = triangle(2*m - 1)
        
        w = []
        for j in range(0,m):
            w.append(c[j]*(4**(1-m)))

        f_hat = []
        for i in range(0,m):
            k = 0 
            for j in range(-m+1,m):
                if i+j >= 0 and i+j < m:
                    k += ssd[i+j]*w[np.abs(j)]
            f_hat.append(k)
        
        
        scaled_freq = []
        for l in lam:
            scaled_freq.append(l/np.pi)
        
        return f_hat, scaled_freq
        
