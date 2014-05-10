import numpy as np

def init_em(calc_kalman,x0,comp_dic):

    #no concentrate parameter (for now)
    #self.conc_param = "none"

    x0, LL_old, dVar = calc_kalman(x0,flag=0) #Initial E-Step and M-Step (E calls M)

    #Replace STAMP initial values with parameter values to initial EM algorithm step
    x0 = [dVar*x for x in x0]
    
    #EM loop
    count = 0
    converged = False
    while not converged:
    
        count  += 1
        x0, LL_curr, dVar = calc_kalman(x0,flag=0) #E-Step and M-Step (E calls M)
        print(x0)
        
        if abs((LL_curr - LL_old)/(LL_old)) < 10**(-6) or count == 150:# or count > 100:
            converged = True
            
            print("Convergence in iteration %i" %count)
            print("loglikelihood: %f" %LL_curr)
            
        LL_old = LL_curr
    
    #Replace initial EM algortithm step parameter values with final EM parameter values
    count = 0
    if "irr" in comp_dic:
        comp_dic["irr"] = x0[count]
        count += 1
    if "level" in comp_dic:
        comp_dic["level"] = x0[count]
        count += 1
    if "slope" in comp_dic:
        comp_dic["slope"] = x0[count]
        count += 1
    if "cycle" in comp_dic:
        comp_dic["cycle"] = x0[count]
        count += 1
    if "seasonal" in comp_dic:
        comp_dic["seasonal"] = x0[count]
        count += 1

         
    #parameter with largest variance from EM step concentrated out
    #when you concentrate out "irr", for example
    #sig_rec is irr variance
    #other params are q = var_param/var_rr
    #thats why we multiply out sig_rec
    conc_val = 0.0
    for var in comp_dic:
        if comp_dic[var] > conc_val:
            conc_val = comp_dic[var]
            conc_param = var
    print("Concentrating out %s" %conc_param)
    

    #adjust init_list to remove concentrated component
    #if there is a concentrate parameter then adjust params
    #to be q ratios
    #if not use params as is
    init_list = []
    if "irr" in comp_dic:
        if conc_param != "irr":
            if conc_param == "none":
                init_list.append(comp_dic["irr"])
            else:
                init_list.append(comp_dic["irr"]/comp_dic[conc_param])
    if "level" in comp_dic:
        if conc_param != "level":
            if conc_param == "none":
                init_list.append(comp_dic["level"])
            else:
                init_list.append(comp_dic["level"]/comp_dic[conc_param])
    if "slope" in comp_dic:
        if conc_param != "slope":
            if conc_param == "none":
                init_list.append(comp_dic["slope"])
            else:
                init_list.append(comp_dic["slope"]/comp_dic[conc_param])
    if "cycle" in comp_dic:
        if conc_param != "cycle":
            if conc_param == "none":
                init_list.append(comp_dic["cycle"])
            else:
                init_list.append(comp_dic["cycle"]/comp_dic[conc_param])
    if "seasonal" in comp_dic:
        if conc_param != "seasonal":
            if conc_param == "none":
                init_list.append(comp_dic["seasonal"])
            else:
                init_list.append(comp_dic["seasonal"]/comp_dic[conc_param])
      
      

    #transform comp_dic values to ensure positive variance
    init_list = [0.5*np.log(i**2) for i in init_list]
    #watch out for fixed params
    for i in range(0,len(init_list)):
        if init_list[i] == -float('Inf'):
            init_list[i] = 0.0        
    if "cycle" in comp_dic:
        if "cycle" in var_list:
            init_list.append(damp)
            init_list.append(freq)


    print(init_list)

    return init_list, conc_param




def init_univar():

    #Initialize #2
       zfix_list = fix_list
       zvar_list = var_list

       self.conc_param = "none"
       res_dic = {}
       for var in comp_dic:
           if var in zvar_list:
           
               self.fix_list = []
               self.var_list = []
               
               x0 = []
               if var == "irr":
                   x0.append(-1.0)
                   self.var_list.append(var)
               else:
                   self.fix_list.append("irr")
               
               if var == "level":
                   x0.append(-0.5)
                   self.var_list.append(var)
               else:
                   self.fix_list.append("level")
                   
               if var == "slope":
                   x0.append(-1.5)
                   self.var_list.append(var)
               else:
                   self.fix_list.append("slope")
                   
               if var == "cycle":
                   self.var_list.append(var)
                   x0.append(-1.5)
                   #d = 0.9
                   #x0.append( np.sqrt( (d**2)/(1-(d**2)) ) )
                   #p = 15
                   #f = np.log(p - 2)
                   #x0.append(f)

               else:
                   self.fix_list.append("cycle")
               
               flag=[1]
               
               res = minimize(calc_kalman, x0, flag, method='BFGS', options={'gtol': 1e-3,'eps': 1e-3,'disp': True, 'maxiter': 5000}) 
           
               if var == "cycle":
                   #damp = (np.abs(res.x[1])*(1+(res.x[1]**2))**(-0.5))
                   #freq = (2.0*np.pi/(2.0 + np.exp(res.x[2])))
                   #damp = res.x[1]
                   #freq = res.x[2]
                   
                   
                   res_dic[var] = float(np.exp(res.x[0])) 
                   #res_dic[var] = res.x[0]*(1.0 - damp**2)
                   flag=[1]
                   
                   
                   
               else:
                   res_dic[var] = float(np.exp(res.x))

           else:
               res_dic[var] = 0.0

       
       damp = 0.9
       freq = 2.0*np.pi/14.0
       self.fix_list = zfix_list
       self.var_list = zvar_list

       x0 = []
       if "irr" in res_dic: 
           x0.append(res_dic["irr"])
       if "level" in res_dic: 
           x0.append(res_dic["level"])
       if "slope" in res_dic: 
           x0.append(res_dic["slope"])
       if "cycle" in res_dic: 
           x0.append(res_dic["cycle"])
