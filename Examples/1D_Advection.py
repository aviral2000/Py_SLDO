import numpy as np    
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from petsc4py import PETSc
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'large',
         'axes.titlesize':'large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

#%% Run scalar-advection problem and generate data 
# Parameters
L = 10.0  # Length of the rod
T = 20.0  # Total time
npts = 201  # Number of spatial grid points
nsteps = 500 # Number of time steps
alpha = 0.0  #0.02 #0.3 # Diffusion coefficient
c = 1.25      #0.2 #1.25
dx = L / (npts - 1)  # Spatial step size
dt = T / nsteps  # Time step size
r1 = alpha * dt / (dx ** 2)
r2 = c*dt/dx
iBC = 1

# Set initial condition
x = dx*np.arange(npts)
# uold = np.sin(5 * np.pi * np.linspace(0, 1, npts))
uinit = np.exp(-(x-3)*(x-3))
params = {}
params['r1'] = r1
params['r2'] = r2
params['npts'] = npts
params['nsteps'] = nsteps
params['iBC'] = iBC
params['dt'] = dt

def solvePDE(params,uinit):
    npts = params['npts']
    nsteps = params['nsteps']
    r1 = params['r1']
    r2 = params['r2']
    iBC = params['iBC']
    dt = params['dt']
    uold = uinit
    u = np.zeros([npts, nsteps])
    ut_data = np.zeros([npts,nsteps])
    # Apply finite difference method
    for t in range(0, nsteps):
        if(iBC==0):
            indx = np.arange(1,npts-1)
            indxp1 = indx+1
            indxm1 = indx-1
        elif(iBC==1):
            indx = np.arange(0,npts)
            indxp1 = indx+1
            indxp1[npts-1] = 0
            indxm1 = indx-1
            indxm1[0] = npts-1
    
        deltau1 = uold[indxp1] - 2 * uold[indx] + uold[indxm1]
        deltau2 = uold[indx] - uold[indxm1]
        u[indx, t] = uold[indx] + r1 * deltau1 - r2 * deltau2
        ut_data[:,t] = (u[:,t] - uold)/dt        
        uold = u[:,t]
    
    return(u, ut_data)

[u_data, ut_data] = solvePDE(params,uinit)

#%% Create animation
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla() # clear the previous image
    ax.plot(x[:], u_data[:,i],label="True") # plot the line
    ax.set_ylim([-1.5, 1.5]) # fix the y axis

from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, frames = range(0,u_data.shape[1],10), interval = 1, 
                               blit = False)

plt.show()


#%% Set BC index function for regression

def setBCindex(iBC, k, nwidth, npts):
    if(iBC == 1):
        if(k<nwidth):
            inda = range(0,k+nwidth+1)
            indb = range(npts-(nwidth-k),npts)
            indx = list(indb) + list(inda)      
        elif(k>npts-(nwidth+1)):
            inda = range(k-nwidth,npts)
            indb = range(0,(k+nwidth+1)-npts)
            indx = list(inda) + list(indb)
        else:
            indx = range(k-nwidth,k+nwidth+1)
    elif(iBC == 0):
        if(k<nwidth1):
            inda = range(0,k+nwidth+1)
            indx = list(inda)      
        elif(k>npts-(nwidth1+1)):
            inda = range(k-nwidth,npts)
            indx = list(inda)
        else:
            indx = range(k-nwidth,k+nwidth+1)
            
    return(indx)

#%% Setup a regression function

def funcTomin(x,Amat,bvec):
    # print("Function:", x, np.sum(np.square(np.matmul(Amat,x) - bvec)))
    return(np.sum(np.square(np.matmul(Amat,x) - bvec)))

def funcTomin_der(x,Amat,bvec):
    tmp = np.matmul(Amat,x) - bvec
    der = 2*np.matmul(Amat.T,tmp)
    # print("Derivative:", der)
    return(der)

def LearnDifferentialOperator(params,iplot,isave,u_data,ut_data):
    nsteps = u_data.shape[1]
    npts = u_data.shape[0]
    iBC = params['iBC']
    learntype = params['learntype']
    nwidth1 = params['nwidthAdvec']
    nwidth2 = params['nwidthDiff']
    epsConst1 = params['epsConst1']
    epsConst2 = params['epsConst2']
    ieqtype = params['ieqtype']
    lda = params['lda']
    errorTrain = np.zeros(npts)
    figname1 = "StabilityPlot_Advec_" + "nwidth1_" + str(nwidth1) + "_nwidth2_" + str(nwidth2) + "_reg_" + str(lda) + ".pdf"
    figname2 = "StabilityPlot_Diff_" + "nwidth1_" + str(nwidth1) + "_nwidth2_" + str(nwidth2) + "_reg_" + str(lda) + ".pdf"
    totalwidth1 = (2*nwidth1)+1
    totalwidth2 = (2*nwidth2)+1
    totalwidth = totalwidth1 + totalwidth2
    advecOpt = np.zeros([npts,npts])
    diffOpt = np.zeros([npts,npts])
    # x0 = np.zeros([totalwidth])
    x0 = np.ones([totalwidth])
    output = {}
    for k in range(0,npts):
        norm = 1.
        bvec = -ut_data[k,:]/norm
        Amat = np.zeros([nsteps,totalwidth])
        indx1 = setBCindex(iBC, k, nwidth1, npts)
        indx2 = setBCindex(iBC, k, nwidth2, npts)        
        print(k,indx1,indx2)
        for i in range(nsteps):
            Amat[i,0:totalwidth1] = c*u_data[indx1,i]/norm
            Amat[i,totalwidth1:totalwidth] = alpha*u_data[indx2,i]/norm
                    
        if(learntype == "Ridge"):    
            LHS = np.matmul(Amat.T,Amat) + lda*np.eye(totalwidth)
            RHS = np.matmul(Amat.T,bvec)
            coefftmp = np.linalg.solve(LHS,RHS)
            val = coefftmp
            advecOpt[indx1,k] = coefftmp[0:totalwidth1]
            diffOpt[indx2,k] = coefftmp[totalwidth1:totalwidth]            
        elif(learntype == "Stable-C"):
                                    
            if(ieqtype == 1):
                ineq_cons = {'type': 'ineq',
                          'fun' : lambda x: lda*np.array([x[nwidth1] - np.sum(np.abs(x[0:nwidth1])) - 
                                                      np.sum(np.abs(x[nwidth1+1:totalwidth1])) - epsConst1]),
                          'jac' : lambda x: lda*np.array(np.concatenate([-1*np.sign(x[0:nwidth1]),[1],
                                                                    -1*np.sign(x[nwidth1+1:totalwidth1]),np.zeros(totalwidth2)]))}
            elif(ieqtype == 2):
                ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: lda*np.array([x[totalwidth1+nwidth2] - np.sum(np.abs(x[totalwidth1:totalwidth1+nwidth2])) - 
                                                                                    np.sum(np.abs(x[totalwidth1+nwidth2+1:totalwidth])) - epsConst2]),
                         'jac' : lambda x: lda*np.array(np.concatenate([np.zeros(totalwidth1),-1*np.sign(x[totalwidth1:totalwidth1+nwidth2]),[1],
                                                                    -1*np.sign(x[totalwidth1+nwidth2+1:totalwidth])]))}        
            else:
                ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: lda*np.array([x[nwidth1] - np.sum(np.abs(x[0:nwidth1])) - 
                                                     np.sum(np.abs(x[nwidth1+1:totalwidth1])) - epsConst1,
                                                        x[totalwidth1+nwidth2] - np.sum(np.abs(x[totalwidth1:totalwidth1+nwidth2])) - 
                                                                                    np.sum(np.abs(x[totalwidth1+nwidth2+1:totalwidth])) - epsConst2]),
                         'jac' : lambda x: lda*np.array([np.concatenate([-1*np.sign(x[0:nwidth1]),[1],
                                                                    -1*np.sign(x[nwidth1+1:totalwidth1]),np.zeros(totalwidth2)]),
                                                        np.concatenate([np.zeros(totalwidth1),-1*np.sign(x[totalwidth1:totalwidth1+nwidth2]),[1],
                                                                    -1*np.sign(x[totalwidth1+nwidth2+1:totalwidth])])])}        
    
            res = minimize(funcTomin, x0, args=(Amat,bvec), method='SLSQP', jac=funcTomin_der,
                            constraints=[ineq_cons], options={'ftol':1e-6, 'disp':True, 'maxiter': 10000})
            val = res.x
            advecOpt[indx1,k] = res.x[0:totalwidth1]
            diffOpt[indx2,k] = res.x[totalwidth1:totalwidth]
            x0 = res.x
        
        errorTrain[k] = funcTomin(val,Amat,bvec) 
        print(funcTomin(val,Amat,bvec))

    [ldaStabAdvec,vec] = np.linalg.eig(-advecOpt)
    [ldaStabDiff,vec] = np.linalg.eig(-diffOpt)
    output['ldaStabAdvec'] = ldaStabAdvec
    output['ldaStabDiff'] = ldaStabDiff
    if(iplot == 1):        
        fig, ax = plt.subplots()  
        ax.scatter(np.real(ldaStabAdvec), np.imag(ldaStabAdvec), marker=".")
        ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
        # ax.legend(loc="upper right")
        ax.set_xlabel("$Re(\lambda)$")
        ax.set_ylabel("$Im(\lambda)$")
        # ax.set_ylim([-8,8])
        if(learntype == "Stable-C"):
            ax.set_xlim([1.1*np.min(np.real(ldaStabAdvec)),-1.1*np.min(np.real(ldaStabAdvec))])
        else:
            ax.set_xlim([-1.1*np.max(np.real(ldaStabAdvec)),1.1*np.max(np.real(ldaStabAdvec))])
        
        if(isave == 1):
            plt.savefig(outpath+figname1, format='pdf')
    elif(iplot == 2):                
        fig, ax = plt.subplots()  
        ax.scatter(np.real(ldaStabDiff), np.imag(ldaStabDiff), marker=".")
        ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
        # ax.legend(loc="upper right")
        ax.set_xlabel("$Re(\lambda)$")
        ax.set_ylabel("$Im(\lambda)$")
        # ax.set_ylim([-8,8])
        ax.set_xlim([1.1*np.min(np.real(ldaStabDiff)),-1.1*np.min(np.real(ldaStabDiff))])
        
        if(isave == 1):
            plt.savefig(outpath+figname2, format='pdf')                        
    elif(iplot == 3):
        fig, ax = plt.subplots()  
        ax.scatter(np.real(ldaStabAdvec), np.imag(ldaStabAdvec), marker=".")
        ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
        ax.set_xlabel("$Re(\lambda)$")
        ax.set_ylabel("$Im(\lambda)$")
        if(learntype == "Stable-C"):
            ax.set_xlim([1.1*np.min(np.real(ldaStabAdvec)),-1.1*np.min(np.real(ldaStabAdvec))])
        else:
            ax.set_xlim([-1.1*np.max(np.real(ldaStabAdvec)),1.1*np.max(np.real(ldaStabAdvec))])
        if(isave == 1):
            plt.savefig(outpath+figname1, format='pdf')
        
        fig, ax = plt.subplots()  
        ax.scatter(np.real(ldaStabDiff), np.imag(ldaStabDiff), marker=".")
        ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
        ax.set_xlabel("$Re(\lambda)$")
        ax.set_ylabel("$Im(\lambda)$")
        ax.set_xlim([1.1*np.min(np.real(ldaStabDiff)),-1.1*np.min(np.real(ldaStabDiff))])
        if(isave == 1):
            plt.savefig(outpath+figname2, format='pdf')    
    
    print(errorTrain.shape)
    output['advecOpt'] = advecOpt
    output['diffOpt'] = diffOpt
    output['errorTrain'] = errorTrain
    return(output)

#%% Obtain operators from data - Paper result: Stability plot - Ridge regression stability plots
iplot = 1
isave = 1
params = {}
params['learntype'] = "Ridge"   ## Ridge, Lasso, Elastic-Net, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [1]   ##[2,3,5,10,20,40], [2]
reglist = [1e-3]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1
errorTrain = np.zeros([npts,len(reglist),len(nwidth1list),len(nwidth2list)])
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthAdvec'] = nwidth1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            errorTrain[:,countlda,countnw1,countnw2] = output['errorTrain']
            
            
#%% Obtain operators from data - Paper result: Stability plot - Ridge regression stability plots (Testing to see if feasible - delete it if not used in paper)
iplot = 1
isave = 0
params = {}
params['learntype'] = "Ridge"   ## Ridge, Lasso, Elastic-Net, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [1]   ##[2,3,5,10,20,40], [2]
reglist = [1e-3]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1
errorTrain = np.zeros([npts,len(reglist),len(nwidth1list),len(nwidth2list)])
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    fig, ax = plt.subplots()     
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthAdvec'] = nwidth1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            ldaStab = output['ldaStabAdvec']                                   
            ax.scatter(np.real(ldaStab), np.imag(ldaStab), marker=".", label=r"$s_l$ = " + str(2*nwidth1+1))

    ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
    ax.legend(loc="upper left")
    ax.set_xlabel("$Re(\lambda)$")
    ax.set_ylabel("$Im(\lambda)$")
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim([1.1*np.min(np.real(ldaStab)),-1.1*np.min(np.real(ldaStab))])          
    
    
#%% Obtain operators from data - Paper result: Stability plot - Stable-C regression stability plot compare stencil size
iplot = 1
isave = 0
params = {}
params['learntype'] = "Stable-C"   ## Ridge, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [1]   ##[2,3,5,10,20,40], [2]
reglist = [1]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1
errorTrain = np.zeros([npts,len(reglist),len(nwidth1list),len(nwidth2list)])
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    fig, ax = plt.subplots()     
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthAdvec'] = nwidth1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            ldaStab = output['ldaStabAdvec']                                   
            ax.scatter(np.real(ldaStab), np.imag(ldaStab), marker=".", label=r"$s_l$ = " + str(2*nwidth1+1))

    ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
    ax.legend(loc="upper left")
    ax.set_xlabel("$Re(\lambda)$")
    ax.set_ylabel("$Im(\lambda)$")
    ax.set_xlim([1.1*np.min(np.real(ldaStab)),-1.1*np.min(np.real(ldaStab))])
    # ax.set_xscale('log')

#%% Obtain operators from data - Paper result: Stability plot - Ridge regression regularization compare plot
iplot = 0
isave = 0
params = {}
params['learntype'] = "Ridge"   ## Ridge, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,5]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [2]   ##[2,3,5,10,20,40], [2]
reglist = [1,1e-1,1e-2,1e-3,1e-4,1e-5]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
errorTrain = np.zeros([npts,len(reglist),len(nwidth1list),len(nwidth2list)])
countnw1 = -1
for nwidth1 in nwidth1list:
    countnw1 = countnw1 + 1
    fig, ax = plt.subplots()     
    countlda = -1
    for lda in reglist:
        countlda = countlda + 1
        params['lda'] = lda
        params['nwidthAdvec'] = nwidth1
        countnw2 = -1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            errorTrain[:,countlda,countnw1,countnw2] = output['errorTrain']
            ldaStab = output['ldaStabAdvec']                                   
            ax.scatter(np.real(ldaStab), np.imag(ldaStab), marker=".", label=r"$\beta_1$ = " + str(lda))

    ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
    ax.legend(loc="upper left")
    ax.set_xlabel("$Re(\lambda)$")
    ax.set_ylabel("$Im(\lambda)$")
    ax.set_xlim([1e-4,2*np.max(np.real(ldaStab))])
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xscale('log')
                            

#%% Obtain operators from data - Paper result: Error plot - Ridge regression regularization compare plot
iplot = 0
isave = 0
params = {}
params['learntype'] = "Ridge"   ## Ridge, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [2]   ##[2,3,5,10,20,40], [2]
reglist = [1,1e-1,1e-2,1e-3,1e-4,1e-5]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
errorTrain = np.zeros([npts,len(reglist),len(nwidth1list),len(nwidth2list)])
countnw1 = -1
fig, ax = plt.subplots()     
for nwidth1 in nwidth1list:
    countnw1 = countnw1 + 1
    countlda = -1
    errorPlot = np.zeros(len(reglist))
    for lda in reglist:
        countlda = countlda + 1
        params['lda'] = lda
        params['nwidthAdvec'] = nwidth1
        countnw2 = -1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            errorTrain[:,countlda,countnw1,countnw2] = output['errorTrain']
        errorPlot[countlda] = np.linalg.norm(errorTrain[:,countlda,countnw1,countnw2])

    ax.plot(reglist,errorPlot,marker=".",label=r"$s_l = $" + str(2*nwidth1+1))

ax.legend(loc="upper left")
ax.set_ylabel(r"$e_{\text{train}}$")
ax.set_xlabel(r"$\beta_1$")
ax.set_yscale('log')
ax.set_xscale('log')

#%% Obtain operators from data - Paper result: Error plot - Stable-C regression plot
iplot = 0
isave = 0
params = {}
params['learntype'] = "Stable-C"   ## Ridge, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [2]   ##[2,3,5,10,20,40], [2]
reglist = [1]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
errorTrain = np.zeros([npts,len(reglist),len(nwidth1list),len(nwidth2list)])
countnw1 = -1
fig, ax = plt.subplots()     
for nwidth1 in nwidth1list:
    countnw1 = countnw1 + 1
    countlda = -1
    errorPlot = np.zeros(len(reglist))
    for lda in reglist:
        countlda = countlda + 1
        params['lda'] = lda
        params['nwidthAdvec'] = nwidth1
        countnw2 = -1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            errorTrain[:,countlda,countnw1,countnw2] = output['errorTrain']
        errorPlot[countlda] = np.linalg.norm(errorTrain[:,countlda,countnw1,countnw2])

    ax.plot(reglist,errorPlot,marker=".",label=r"$s_l = $" + str(nwidth1))

ax.legend(loc="upper left")
ax.set_ylabel(r"$e_{\text{train}}$")
ax.set_xlabel(r"$\beta_1$")
ax.set_yscale('log')
ax.set_xscale('log')


#%% Paper result: Determine operators and propagate them through equation solution
fac = 1
iStart = 0
dt_new = dt/fac
nsteps2 = 3*int(T/dt_new)

paramsPDE = {}
paramsPDE['r1'] = r1
paramsPDE['r2'] = r2
paramsPDE['npts'] = npts
paramsPDE['nsteps'] = nsteps2
paramsPDE['iBC'] = iBC
paramsPDE['dt'] = dt

uinit2 = uinit
# uinit2 = np.exp(-(x-3)*(x-3)/0.1) + 0.1*np.random.randn(npts)
                               
def solvePDE_Opt(params,uinit,advecOpt,diffOpt):
    nsteps = params['nsteps']
    rfac1 = params['rfac1']
    rfac2 = params['rfac2']
    u_new =  np.zeros([npts,nsteps])
    uold = uinit
    for t in range(0, nsteps):
        deltau1 = -np.matmul(uold,diffOpt)
        deltau2 = -np.matmul(uold,advecOpt)
        u_new[:, t] = uold + rfac1 * deltau1 + rfac2 * deltau2
        uold = u_new[:,t]     
    return(u_new)

paramsPDE_Opt = {}
paramsPDE_Opt['rfac1'] = dt_new*alpha
paramsPDE_Opt['rfac2'] = dt_new*c
paramsPDE_Opt['nsteps'] = nsteps2
     
iplot = 0
isave = 0
params = {}
params['learntype'] = "Ridge"   ## Ridge, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [2]   ##[2,3,5,10,20,40], [2]
reglist = [1e-3]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1

[u_data_PDE, tmp] = solvePDE(paramsPDE,uinit2)

u_data_PDE_Ridge = np.zeros([npts,nsteps2,len(reglist),len(nwidth1list),len(nwidth2list)])

outputRidge = {}
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    outputRidge[str(lda)] = {}
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthAdvec'] = nwidth1
        outputRidge[str(lda)][str(nwidth1)] = {}
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            outputRidge[str(lda)][str(nwidth1)][str(nwidth2)] = output
            advecOpt = output['advecOpt']
            diffOpt = output['diffOpt']
            u_data_PDE_Ridge[:,:,countlda,countnw1,countnw2] = solvePDE_Opt(paramsPDE_Opt,uinit2,advecOpt,diffOpt)
            

#%% Also compute stable-C solution and store it differently 
paramsPDE_Opt = {}
paramsPDE_Opt['rfac1'] = dt_new*alpha
paramsPDE_Opt['rfac2'] = dt_new*c
paramsPDE_Opt['nsteps'] = nsteps2
     
iplot = 0
isave = 0
params = {}
params['learntype'] = "Stable-C"   ## Ridge, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['iBC'] = iBC
nwidth1list = [1,2,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [1]   ##[2,3,5,10,20,40], [2]
reglist = [1]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1

u_data_PDE_StableC = np.zeros([npts,nsteps2,len(reglist),len(nwidth1list),len(nwidth2list)])

outputStableC = {}
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    outputStableC[str(lda)] = {}
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthAdvec'] = nwidth1
        outputStableC[str(lda)][str(nwidth1)] = {}
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            outputStableC[str(lda)][str(nwidth1)][str(nwidth2)] = output
            advecOpt = output['advecOpt']
            diffOpt = output['diffOpt']
            u_data_PDE_StableC[:,:,countlda,countnw1,countnw2] = solvePDE_Opt(paramsPDE_Opt,uinit2,advecOpt,diffOpt)
            
        
#%% Compute average coefficients over the entire domain
advecOptavgRidge = {}
diffOptavgRidge = {}
reglist = [1e-3]    
nwidth1list = [1,2,3,5,10,20] 
nwidth2list =  [2]
for lda in reglist:
    advecOptavgRidge[str(lda)] = {}
    diffOptavgRidge[str(lda)] = {}
    for nwidth1 in nwidth1list:
        advecOptavgRidge[str(lda)][str(nwidth1)] = {}
        diffOptavgRidge[str(lda)][str(nwidth1)] = {}
        for nwidth2 in nwidth2list:
            outputAdvec = outputRidge[str(lda)][str(nwidth1)][str(nwidth2)]['advecOpt']
            outputDiff  = outputRidge[str(lda)][str(nwidth1)][str(nwidth2)]['diffOpt']
            advectmp = 0.
            difftmp = 0.
            for k in range(0,npts):
                indx1 = setBCindex(iBC, k, nwidth1, npts)
                indx2 = setBCindex(iBC, k, nwidth2, npts)  
                advectmp = advectmp + outputAdvec[indx1,k]
                difftmp = difftmp + outputDiff[indx2,k]        
            advecOptavgRidge[str(lda)][str(nwidth1)][str(nwidth2)] = advectmp*dx/npts
            diffOptavgRidge[str(lda)][str(nwidth1)][str(nwidth2)] = difftmp*dx/npts
            
            
advecOptavgStableC = {}
diffOptavgStableC = {}
reglist = [1]    
nwidth1list = [1,2,3,5,10,20] 
nwidth2list =  [2]
for lda in reglist:
    advecOptavgStableC[str(lda)] = {}
    diffOptavgStableC[str(lda)] = {}
    for nwidth1 in nwidth1list:
        advecOptavgStableC[str(lda)][str(nwidth1)] = {}
        diffOptavgStableC[str(lda)][str(nwidth1)] = {}
        for nwidth2 in nwidth2list:
            outputAdvec = outputStableC[str(lda)][str(nwidth1)][str(nwidth2)]['advecOpt']
            outputDiff  = outputStableC[str(lda)][str(nwidth1)][str(nwidth2)]['diffOpt']
            advectmp = 0.
            difftmp = 0.
            for k in range(0,npts):
                indx1 = setBCindex(iBC, k, nwidth1, npts)
                indx2 = setBCindex(iBC, k, nwidth2, npts)  
                advectmp = advectmp + outputAdvec[indx1,k]
                difftmp = difftmp + outputDiff[indx2,k]        
            advecOptavgStableC[str(lda)][str(nwidth1)][str(nwidth2)] = advectmp*dx/npts
            diffOptavgStableC[str(lda)][str(nwidth1)][str(nwidth2)] = difftmp*dx/npts            
            
            
#%% Compare solutions at certain time-step for different nwidth1

cp = ['blue','orange','green','red','purple','brown']
tsteplist = [10,20,500]
for ts in tsteplist:
    fig, ax = plt.subplots()
    ax.plot(x,u_data_PDE[:,ts],label="Reference",color="black")
    for i in range(len(nwidth1list)):         
        sl = str(2*nwidth1list[i]+1)
        ax.plot(x,u_data_PDE_Ridge[:,ts,0,i,0],label="LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])
        # ax.plot(x,u_data_PDE_StableC[:,ts,0,0,i],label="S-LDO: $s_l = $" + str(sl) ,linestyle='solid', color=cp[i])
        
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_ylim([-0.1,1.4])
    
cp = ['blue','orange','green','red','purple','brown']
for ts in tsteplist:
    fig, ax = plt.subplots()
    ax.plot(x,u_data_PDE[:,ts],label="Reference",color="black")
    for i in range(len(nwidth1list)):         
        sl = str(2*nwidth1list[i]+1)
        # ax.plot(x,u_data_PDE_Ridge[:,ts,0,0,i],label="LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])
        ax.plot(x,u_data_PDE_StableC[:,ts,0,i,0],label="S-LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])
        
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_ylim([-0.1,1.4])    
    
    
#%% Compare errors in time
tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
normval = np.linalg.norm(u_ref,axis=0)
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    errorRidge = np.linalg.norm(u_data_PDE_Ridge[:,:,0,i,0] - u_ref,axis=0)
    errorStableC = np.linalg.norm(u_data_PDE_StableC[:,:,0,i,0] - u_ref,axis=0)
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorRidge/normval,label="LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])

ax.axvspan(0.0, dt*nsteps, alpha=0.2, color='red')
ax.legend(loc="upper right")
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([-0.1,1])

tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    errorRidge = np.linalg.norm(u_data_PDE_Ridge[:,:,0,i,0] - u_ref,axis=0)
    errorStableC = np.linalg.norm(u_data_PDE_StableC[:,:,0,i,0] - u_ref,axis=0)
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorStableC/normval,label="S-LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])

ax.axvspan(0.0, dt*nsteps, alpha=0.2, color='red')
ax.legend()
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([-0.1,1])
    

#%% Create animation
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla() # clear the previous image
    ax.plot(x[:], u_data_PDE[:,i],label="Reference") # plot the line
    ax.plot(x[:], u_data_PDE_Ridge[:,i,0,0,0],linestyle="dashed",label="LDO") # plot the line
    ax.plot(x[:], u_data_PDE_StableC[:,i,0,0,0],linestyle="dashed",label="S-LDO") # plot the line

    # ax.set_xlim([, tfinal]) # fix the x axis
    ax.set_ylim([-0.5, 2.0]) # fix the y axis

from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, frames = range(0,nsteps2,20*fac), interval = 1, 
                               blit = False)

plt.show()
# anim.save('/Users/aviral/Desktop/1Dadvection.mp4')

#%% FD stencil linear stability plot

nwidthtmp = 1
MatFD = np.zeros([npts,npts])
MatCD = np.zeros([npts,npts])
for k in range(npts):
    ind = setBCindex(iBC, k, nwidthtmp, npts)
    MatFD[k,ind] = np.array([-1.00,1.00,0])/dx
    MatCD[k,ind] = np.array([0,-1,1])/(dx)

[ldatmp,vec] = np.linalg.eig(-MatFD)
fig, ax = plt.subplots()  
ax.scatter(np.real(ldatmp), np.imag(ldatmp), marker=".")
ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
ax.set_xlabel("$Re(\lambda)$")
ax.set_ylabel("$Im(\lambda)$")
ax.set_xlim([-45,45])

[ldatmp,vec] = np.linalg.eig(-MatCD)
fig, ax = plt.subplots()  
ax.scatter(np.real(ldatmp), np.imag(ldatmp), marker=".")
ax.axvspan(-1000, 0.0, alpha=0.2, color='red')
ax.set_xlabel("$Re(\lambda)$")
ax.set_ylabel("$Im(\lambda)$")
ax.set_xlim([-45,45])

