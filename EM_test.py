#import modules
from pandas import DataFrame
import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt


def kalman(x0,y,d):
            n = len(y)
        
            #create static matrices: Z, R, T
            Z = np.zeros(d)
            Z[0] = 1
            Z = np.matrix(Z)
            R = np.matrix(np.identity(d))        
            #T = np.matrix(np.identity(d))
            
            T = np.matrix(np.identity(d))
            if "slope" in var_list: 
                T[0,1] = 1  
            
            ##!!! T!
            
            D = np.matrix([[0]])
            C = np.matrix([[0]])
            
            count = 0
            if "irr" in var_list:
                q_irr = x0[count]
                count += 1
            else:
                q_irr = 0  
            if "level" in var_list:
                q_level = x0[count]
                count += 1
            else:
                q_level = 0
            if "slope" in var_list:
                q_slope = x0[count]  
                count += 1
            else:
                q_slope = 0 

            
            H = q_irr

            Q = np.matrix(np.identity(d))
            Q[0,0] = q_level

            if "slope" in var_list:
                Q[1,1] = q_slope
            

            
            #create matrix of initial state levels
            a0 = 0
            a0 = [a0 for i in range(d)]
            a0[0] = np.mean(y)
            a0 = np.matrix(a0)
            a0 = a0.T
            A = [a0]
                
            #create matrix of initial state variances (diffuse prior)
            p0 = 1000
            p0 = np.matrix(np.identity(d))*p0
            P = [p0]
                
            #create expected state level and expected state variance lists
            EA = []
            EP = []
            
            #create 1 step ahead forecast error (V) 
            #and 1 step ahead forecast error variance (F) lists
            #each V also known as an innovation
            V = []
            F = []
            
            #other matrices required by kalman function
            epsilon = []
            L = []
            K = []
            LL_term2_rec = 0
            LL_term3_rec = 0
        
            sig_rec = 0
        
            #start kalman recurision
            for t in range(0,n):          

                #prediction equations
                EA.append(T*A[t] + C)
                EP.append(T*P[t]*T.T + R*Q*R.T)
                    
                #likelihood equations
                V.append(y[t] - Z*EA[t] - D)

                F.append(Z*EP[t]*Z.T + H)
                   
                #updating equations
                A.append(EA[t] + EP[t]*Z.T*np.linalg.pinv(F[t])*V[t])
                P.append(EP[t] - EP[t]*Z.T*np.linalg.pinv(F[t])*Z*EP[t])

                #calculate K           
                K.append(  T*EP[t]*Z.T*np.linalg.pinv(F[t])  ) 
                
                #calculate L           
                L.append(  T - K[t]*Z  )  
                       
                #Standardised Prediction Errors
                #page 54 Koopman
                epsilon.append(V[t]*np.linalg.pinv(np.sqrt(F[t])))
                            
                #Likelihood Function
                #page 53 Koopman
                #from d+1
                #subtract 1 for zero indexing
                if t >= d + 1 - 1:
                    #conc
                    sig_curr = V[t].T*np.linalg.pinv(F[t])*V[t]
                    sig_rec = sig_rec + sig_curr
                
                    LL_term2_curr = np.log(np.linalg.det(F[t]))
                    LL_term2_rec = LL_term2_rec + LL_term2_curr
                    
                    LL_term3_curr = V[t].T*np.linalg.pinv(F[t])*V[t]
                    LL_term3_rec = LL_term3_rec + LL_term3_curr
                else:
                    LL_term2_rec = 0
                    LL_term3_rec = 0
            
            dVar = (1/float(n-1))*LL_term3_rec
            
            sig_rec = (1/float(n - 1))*sig_rec
            sig1 = -(n/2)*np.log(2*np.pi)
            sig2 = -((n-1)/2.0)
            sig3 = -((n-1)/2.0)*np.log(sig_rec)
            sig4 = -0.5*LL_term2_rec
            
            LLc = sig1 + sig2 + sig3 + sig4
            
            LL_term1 = -float(n/2.0)*np.log(2*np.pi)
            
            #LL is the term we want to maximise
            #LL is what we want to know about
            #-LL is only used to change the min function into a max function
            LL = LL_term1 - 0.5*LL_term2_rec - 0.5*LL_term3_rec 
            LL = LL/float(n)
            
            fin_a = A[-1]
            A = A[1:-1]
            
            if d >= 2:
                L2 = np.mat([[np.nan], [np.nan]])
                A.insert(0,L2)
            else:
                L2 = []
                for i in range(1):
                    L2.append(np.nan)
                L2 = np.mat(L2)
                A.insert(0,L2)
            

            jp, fs = smooth(V,F,L,EA,EP,K,H,Q,R,A,P,C,sig_rec,fin_a,Z,T,d)

            return jp, LL, fs, dVar


def smooth(V,F,L,EA,EP,K,H,Q,R,A,P,C,sig_rec,fin_a,Z,T,d):    
        
        n = len(y)
        
        #Classical Fixed Interval Smoothing Algorithm
        #create list of smoothed level
        #may need to seed with what we deleted from A before
        S = [fin_a]
        for t in range(n,-1,-1):
            S.append(A[t-1] + P[t-1] * T.T * np.linalg.pinv(EP[t-1]) * (S[n - t] - T*A[t-1] - C) )
        cfis = S
        cfis = cfis[:-2]

        #Fixed Interval Smoothing Algorithm
        #r2 is the smoothing state cumulant
        r1 = [np.mat([0]*d).T]
        for t in range(n-1,-1,-1):
            r1.append(  Z.T*np.linalg.pinv(F[t])*V[t] + L[t].T*r1[n-1-t] ) 
        r2 = []
        for e in reversed(r1):
            r2.append(e)
        fis = []
        for t in range(0,n):
            fis.append(EA[t] + EP[t]*r2[t])   
                  
        #Fixed Interval Smoothed State Variance
        #N is the smoothing state variance cumulant 
        
        tF = F[1:]
        tF.insert(0,np.mat([np.nan]))                           
        n1 = [np.zeros((d,d))]
        for t in range(n-1,-1,-1):
            n1.append(  Z.T*np.linalg.pinv(tF[t])*Z + L[t].T*n1[n-1-t]*L[t] ) 
        N = []
        for e in reversed(n1):
        #for e in n1:
            N.append(e)
        
        #NN = [float(nn) for nn in N]

        #tP = P[1:]
        #tP.insert(0,np.mat([np.nan]))
        #tEP = EP[1:]
        #tEP.insert(0,np.mat([np.nan]))
        V2 = []
        for t in range(0,n):
            #we want N(t-1)
            V2.append(EP[t] - EP[t]*N[t]*EP[t])
        #VV = [float(vv) for vv in V2]


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
        
        #Fast Smoothing Algorithm
        fs = [EA[0] + EP[0]*r2[0]]
        for t in range(0,n):
            fs.append(T*fs[t] + R*eta[t])
        fs = fs[:-1]

        B = 0
        c = 0
        for t in range(0,n):
            
            c += eps[t][0]**2 + var_eps[t]
            
            if t >= d:
                B += eta[t-1]*eta[t-1].T + var_eta[t-1]    
        
        xsi_irr = float(c/float(n))
        xsi_level = B[0,0]/float(n-1)
        
        
        if d >= 2:
            xsi_slope = B[1,1]/float(n-1)
            r_x0 = [xsi_irr,xsi_level,xsi_slope]
        else:
            r_x0 =[xsi_irr,xsi_level]
        
        return r_x0, fs
      




#choose file
#file_name = '/Users/joestox/Documents/datasets/koopman-commandeur/ukdrivers2.txt'
file_name = '/Users/joestox/Documents/datasets/koopman-commandeur/norwayfinland.txt'
#file_name = '/Users/joestox/Documents/datasets/Durbin-Koopman-data/nile.txt'


#upload file
try:
    xl = pd.ExcelFile(file_name, na_values=['na', 'NA'])
    sheets = xl.sheet_names
    df = xl.parse(sheets[0])
    print('xls')
except:
    df = DataFrame(pd.read_csv(file_name, na_values=['na', 'NA']))
    print('csv')

#select variable to send to KF algorithm
#y = np.log(df['KSI'])
y = np.log(df['Norwegian_fatalities'])
#y = df['vol']


var_list = ["irr","level","slope"]
x0 = [-1.0,-0.5,-1.5]

#var_list = ["irr","level"]
#x0 = [-1.0,-0.5]

#number of diffuse initial elements in state
d = len(var_list) - 1
    
x0, LL_old, fs, dVar = kalman(x0,y,d) #Initial E-Step and M-Step (E calls M)
x0 = [dVar*x for x in x0] #Initial parameter values



converged = False
count = 0
while not converged:

    count  += 1
    x0, LL_curr, fs, dVar = kalman(x0,y,d) #E-Step and M-Step (E calls M)
    print(x0)
    
    if abs((LL_curr - LL_old)/(LL_old)) < 10**(-12):# or count > 100:
        converged = True
        
        print("Convergence in iteration %i" %count)
        print("loglikelihood: %f" %LL_curr)
        
        print(x0[1]/x0[0])
        if d>=2:
            print(x0[2]/x0[0])
        
        plt.plot(y)
        fs = [float(fsl[0,0]) for fsl in fs]
        plt.plot(fs)
        pylab.show()
           
    LL_old = LL_curr

