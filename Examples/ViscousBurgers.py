import numpy as np    
import matplotlib.pyplot as plt
from petsc4py import PETSc
import math
from scipy.optimize import minimize
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

#%% Solve PDE 

scheme = "explicit"
N = 128 #1024
x0 = 0.
xN = 2*math.pi
h = (xN-x0)/N
xdist = np.arange(N)*h
deltaT  = 0.002
mu = 0.1
T = 3.0
nsteps = int(T/deltaT)
iBC = 1

eps = 2e-1
umean = 0.3
np.random.seed(10)
uinit = np.random.normal(loc=umean,scale=eps,size=N)
# uinit = np.sin(xdist)


paramsPDE = {}
paramsPDE['scheme'] = scheme
paramsPDE['N'] = N
paramsPDE['h'] = h
paramsPDE['deltaT'] = deltaT
paramsPDE['mu'] = mu
paramsPDE['nsteps'] = nsteps
paramsPDE['uinit'] = uinit



def solvePDE(paramsPDE):
    scheme = paramsPDE['scheme']
    N = paramsPDE['N']
    h = paramsPDE['h']
    deltaT = paramsPDE['deltaT']
    mu = paramsPDE['mu']
    nsteps = paramsPDE['nsteps']
    uinit = paramsPDE['uinit']

    def compute_Res_FD(snes, x, res, dt, dx, mu, u_old):
        npts = u_old.shape[0]
        ind = np.arange(npts, dtype=np.int32)
        indm1 = ind-1
        indp1 = ind+1
        indp1[npts-1] = 0
        advec = (1./(2*dx))*u_old[ind]*(x[indp1] - x[indm1])
        # advec = (0.25/dx)*(np.square(x[indp1]) - np.square(x[indm1]))
        diff = (1./(dx**2))*mu*(x[indp1] - 2*x[ind] + x[indm1])
        res[:] = x[ind] - u_old[ind] + dt*(advec - diff)    
        return(res)
    
    commPETSc = PETSc.COMM_WORLD
    res = PETSc.Vec().createSeq(N)
    snes = PETSc.SNES().create(comm=commPETSc)
    
    u_data = np.zeros([N,nsteps])
    ut_data = np.zeros([N,nsteps])
    u_old = uinit
    
    if(scheme == "explicit"):
        npts = u_old.shape[0]
        ind = np.arange(npts, dtype=np.int32)
        indm1 = ind-1
        indp1 = ind+1
        indp1[npts-1] = 0
        for it in range(0, nsteps):
            deltau1 = -u_old[ind]*(u_old[indp1] - u_old[indm1])/(2*h)
            deltau2 = mu*(u_old[indp1] - 2*u_old[ind] + u_old[indm1])/(h**2)
            u_data[:,it] = u_old + deltaT*(deltau1 + deltau2) 
            ut_data[:,it] = (u_data[:,it] - u_old)/deltaT
            u_old = u_data[:,it]
    elif(scheme == "implicit"):
        b = None
        for it in range(nsteps):
            snes.setFunction(compute_Res_FD,res,args=(deltaT,h,mu,u_old))
            x = res.duplicate()
            x[:] = u_old
            snes.solve(b,x)
            u_data[:,it] = x[:]
            ut_data[:,it] = (u_data[:,it] - u_old)/deltaT
            u_old = u_data[:,it]
            if(it%10==0):
                print("Completed, it=",it)
    
    return(u_data, ut_data)

[u_data, ut_data] = solvePDE(paramsPDE)


#%% Plot the results
it = 400
fig, ax = plt.subplots()  
ax.plot(xdist, u_data[:,100], label="True") 
ax.plot(xdist, u_data[:,400], label="True") 

ax.legend(loc="upper right")
# ax.set_ylabel("Error")
# ax.set_xlabel("t")    

#%% Create animation
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla() # clear the previous image
    ax.plot(xdist[:], u_data[:,i],label="True") # plot the line
    ax.set_ylim([-0.5, 1.5]) # fix the y axis

from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, frames = range(0,nsteps,5), interval = 1, 
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
    x0 = np.zeros([totalwidth])
    if(learntype == "Ridge"):
        lda1 = params['lda1']
        lda2 = params['lda2']
        ldaMat = np.zeros([totalwidth,totalwidth])
        for i in range(totalwidth):
            if i < totalwidth1:
                ldaMat[i,i] = lda1
            else:
                ldaMat[i,i] = lda2
    output = {}
    for k in range(0,npts):
        norm = 1.
        bvec = -ut_data[k,:]/norm
        Amat = np.zeros([nsteps,totalwidth])
        indx1 = setBCindex(iBC, k, nwidth1, npts)
        indx2 = setBCindex(iBC, k, nwidth2, npts)        
        print(k,indx1,indx2)
        for i in range(nsteps):
            Amat[i,0:totalwidth1] = u_data[k,i]*u_data[indx1,i]/norm
            # Amat[i,0:totalwidth1] = 0.5*np.square(u_data[indx1,i])
            Amat[i,totalwidth1:totalwidth] = mu*u_data[indx2,i]/norm
                    
        if(learntype == "Ridge"):    
            LHS = np.matmul(Amat.T,Amat) + ldaMat
            RHS = np.matmul(Amat.T,bvec)
            coefftmp = np.linalg.solve(LHS,RHS)
            val = coefftmp
            advecOpt[indx1,k] = coefftmp[0:totalwidth1]
            diffOpt[indx2,k] = coefftmp[totalwidth1:totalwidth]            
        elif(learntype == "Stable-C"):
            
            ## ieqtype defines the type of constraint used                             
            if(ieqtype == 1):
                ineq_cons = {'type': 'ineq',
                          'fun' : lambda x: np.array([x[nwidth1] - np.sum(np.abs(x[0:nwidth1])) - 
                                                      np.sum(np.abs(x[nwidth1+1:totalwidth1])) - epsConst1]),
                          'jac' : lambda x: np.array(np.concatenate([-1*np.sign(x[0:nwidth1]),[1],
                                                                    -1*np.sign(x[nwidth1+1:totalwidth1]),np.zeros(totalwidth2)]))}
            elif(ieqtype == 2):
                ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: np.array([x[totalwidth1+nwidth2] - np.sum(np.abs(x[totalwidth1:totalwidth1+nwidth2])) - 
                                                                                    np.sum(np.abs(x[totalwidth1+nwidth2+1:totalwidth])) - epsConst2]),
                         'jac' : lambda x: np.array(np.concatenate([np.zeros(totalwidth1),-1*np.sign(x[totalwidth1:totalwidth1+nwidth2]),[1],
                                                                    -1*np.sign(x[totalwidth1+nwidth2+1:totalwidth])]))}        
            elif(ieqtype == 3):
                ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: np.array([x[nwidth1] - np.sum(np.abs(x[0:nwidth1])) - 
                                                     np.sum(np.abs(x[nwidth1+1:totalwidth1])) - epsConst1,
                                                        x[totalwidth1+nwidth2] - np.sum(np.abs(x[totalwidth1:totalwidth1+nwidth2])) - 
                                                                                    np.sum(np.abs(x[totalwidth1+nwidth2+1:totalwidth])) - epsConst2]),
                         'jac' : lambda x: np.array([np.concatenate([-1*np.sign(x[0:nwidth1]),[1],
                                                                    -1*np.sign(x[nwidth1+1:totalwidth1]),np.zeros(totalwidth2)]),
                                                        np.concatenate([np.zeros(totalwidth1),-1*np.sign(x[totalwidth1:totalwidth1+nwidth2]),[1],
                                                                    -1*np.sign(x[totalwidth1+nwidth2+1:totalwidth])])])}        
            elif(ieqtype == 4):
                ineq_cons = {'type': 'ineq',
                          'fun' : lambda x: np.array([umean*(2*x[nwidth1] - np.sum(np.abs(x[0:nwidth1])) - 
                                                      np.sum(np.abs(x[nwidth1+1:totalwidth1]))) - epsConst1]),
                          'jac' : lambda x: np.array(np.concatenate([-umean*np.sign(x[0:nwidth1]),[2*umean],
                                                                    -umean*np.sign(x[nwidth1+1:totalwidth1]),np.zeros(totalwidth2)]))}
            elif(ieqtype == 5):
                ineq_cons = {'type': 'ineq',
                         'fun' : lambda x: np.array([umean*(2*x[nwidth1] - np.sum(np.abs(x[0:nwidth1])) - 
                                                     np.sum(np.abs(x[nwidth1+1:totalwidth1]))) - epsConst1,
                                                        x[totalwidth1+nwidth2] - np.sum(np.abs(x[totalwidth1:totalwidth1+nwidth2])) - 
                                                                                    np.sum(np.abs(x[totalwidth1+nwidth2+1:totalwidth])) - epsConst2]),
                         'jac' : lambda x: np.array([np.concatenate([-umean*np.sign(x[0:nwidth1]),[2*umean],
                                                                    -umean*np.sign(x[nwidth1+1:totalwidth1]),np.zeros(totalwidth2)]),
                                                        np.concatenate([np.zeros(totalwidth1),-1*np.sign(x[totalwidth1:totalwidth1+nwidth2]),[1],
                                                                    -1*np.sign(x[totalwidth1+nwidth2+1:totalwidth])])])}   
            elif(ieqtype == 6):  ## used in the article: closest to the theory while removing nonlinear operator terms from the absolute sign for ease of implementation
                mu2 = mu
                ineq_cons = {'type': 'ineq',
                          'fun' : lambda x: np.array([2*umean*x[nwidth1] + mu2*(x[totalwidth1+nwidth2] - np.sum(np.abs(x[totalwidth1:totalwidth1+nwidth2])) - 
                                                      np.sum(np.abs(x[totalwidth1+nwidth2+1:totalwidth]))) - epsConst1]),
                          'jac' : lambda x: np.array([np.concatenate([np.zeros(nwidth1),[2*umean],np.zeros(totalwidth1-nwidth1-1),
                                                                      -1*mu2*np.sign(x[totalwidth1:totalwidth1+nwidth2]),[mu2],
                                                                    -1*mu2*np.sign(x[totalwidth1+nwidth2+1:totalwidth])])])}                   
    
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

        

#%% Solve Burgers based on computed derivatives

fac = 1
scheme = "explicit"    #explicit, implicit
dt_new = deltaT/fac
nsteps2 = int(T/dt_new)
mu2 = mu
# np.random.seed(13)
# uinit2 = np.random.normal(loc=umean,scale=eps,size=N)
uinit2 = uinit

paramsPDE = {}
paramsPDE['scheme'] = scheme
paramsPDE['N'] = N
paramsPDE['h'] = h
paramsPDE['deltaT'] = dt_new
paramsPDE['mu'] = mu2
paramsPDE['nsteps'] = nsteps2
paramsPDE['uinit'] = uinit2


paramsODE_opt = {}
paramsODE_opt['explicit'] = scheme
paramsODE_opt['dt'] = dt_new
paramsODE_opt['nsteps'] = nsteps2
paramsODE_opt['uinit'] = uinit2
paramsODE_opt['N'] = N
paramsODE_opt['h'] = h
paramsODE_opt['mu'] = mu2


def compute_Res_DD(snes, x, res, dt, dx, mu, u_old, advecOpt, diffOpt):
    advecterm = u_old[:]*np.matmul(x[:],advecOpt)
    # advecterm = 0.5*np.matmul(np.square(x[:]),advecOpt)
    diffterm = mu*np.matmul(x[:],diffOpt)
    res[:] = x[:] - u_old[:] + dt*(advecterm + diffterm)    
    return(res)      

def solvePDE_Opt(paramsODE_opt, advecOpt, diffOpt):
    scheme = paramsODE_opt['explicit']
    dt_new = paramsODE_opt['dt']
    nsteps2 = paramsODE_opt['nsteps']
    uinit = paramsODE_opt['uinit']
    N = paramsODE_opt['N']
    h = paramsODE_opt['h']   
    mu = paramsODE_opt['mu']
    
    u_new = np.zeros([N,nsteps2])
    u_old = uinit
    if(scheme == "explicit"):
        for it in range(0, nsteps2):
            deltau1 = -u_old*np.matmul(u_old,advecOpt)
            deltau2 = -mu*np.matmul(u_old,diffOpt)
            u_new[:,it] = u_old + dt_new*(deltau1 + deltau2) 
            u_old = u_new[:,it]
    elif(scheme == "implicit"):
        commPETSc = PETSc.COMM_WORLD
        res = PETSc.Vec().createSeq(N)
        snes = PETSc.SNES().create(comm=commPETSc)
        
        b = None
        for it in range(nsteps2):    
            snes.setFunction(compute_Res_DD,res,args=(dt_new,h,mu,u_old,advecOpt,diffOpt))
            x = res.duplicate()
            x[:] = u_old
            snes.solve(b,x)
            u_new[:,it] = x[:]
            u_old = u_new[:,it]
            if(it%10==0):
                print("Completed, it=",it)   
                
    return(u_new)

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
nwidth1list = [1,2,3,5]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [1,2,3,5]   ##[2,3,5,10,20,40], [2]
reglist = [1e-2]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1

[u_data_PDE, tmp] = solvePDE(paramsPDE)

u_data_PDE_Ridge = np.zeros([N,nsteps2,len(reglist),len(nwidth1list),len(nwidth2list)])

for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda1'] = 0.1
    params['lda2'] = lda
    params['lda'] = lda
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthAdvec'] = nwidth1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthDiff'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            advecOpt = output['advecOpt']
            diffOpt = output['diffOpt']
            u_data_PDE_Ridge[:,:,countlda,countnw1,countnw2] = solvePDE_Opt(paramsODE_opt,advecOpt,diffOpt)
            


#%% Also compute stable-C solution and store it differently 
     
iplot = 0
isave = 0
params = {}
params['learntype'] = "Stable-C"   ## Ridge, Stable-C
params['ieqtype'] = 6
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['iBC'] = iBC
nwidth1list = [1,2,3,5]    ##[1,2,3,5,10,20], [1]
nwidth2list =  [1,2,3,5]   ##[2,3,5,10,20,40], [2]

reglist = [1]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1

u_data_PDE_StableC = np.zeros([N,nsteps2,len(reglist),len(nwidth1list),len(nwidth2list)])

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
            advecOpt = output['advecOpt']
            diffOpt = output['diffOpt']
            u_data_PDE_StableC[:,:,countlda,countnw1,countnw2] = solvePDE_Opt(paramsODE_opt,advecOpt,diffOpt)

#%% Create animation
fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla() # clear the previous image
    ax.plot(xdist, u_data_PDE[:,i],label="True") # plot the line
    ax.plot(xdist, u_data_PDE_Ridge[:,i,0,2,2],linestyle="dashed",label="LDO") # plot the line
    ax.plot(xdist, u_data_PDE_StableC[:,i,0,2,2],linestyle="dashed",label="S-LDO-1") # plot the line
   
    ax.set_ylim([-0.5, 1.5]) # fix the y axis

from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, frames = range(0,nsteps2,5*fac), interval = 1, 
                               blit = False)
# anim = animation.FuncAnimation(fig, animate, frames = range(0,nsteps2,2*fac), interval = 20, 
#                                blit = False)
plt.show()
plt.legend()
# anim.save('/Users/aviral/Desktop/1DBurgers.mp4')


#%% Compare solutions at certain time-step for different nwidth1

cp = ['blue','orange','green','red','purple','brown']
tsteplist = [100,500,1000]
for ts in tsteplist:
    fig, ax = plt.subplots()
    ax.plot(xdist,u_data_PDE[:,ts],label="Reference",color="black")
    for i in range(len(nwidth1list)):         
        sl = str(2*nwidth1list[i]+1)
        ax.plot(xdist,u_data_PDE_Ridge[:,ts,0,i,1],label="LDO: $s_{l_1} = $" + str(sl) ,linestyle='dashed', color=cp[i])
        # ax.plot(x,u_data_PDE_StableC[:,ts,0,0,i],label="S-LDO: $s_l = $" + str(sl) ,linestyle='solid', color=cp[i])
        
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_ylim([0.1,0.65])
    
cp = ['blue','orange','green','red','purple','brown']
tsteplist = [100,500,1000]
for ts in tsteplist:
    fig, ax = plt.subplots()
    ax.plot(xdist,u_data_PDE[:,ts],label="Reference",color="black")
    for i in range(len(nwidth1list)):         
        sl = str(2*nwidth1list[i]+1)
        # ax.plot(x,u_data_PDE_Ridge[:,ts,0,0,i],label="LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])
        ax.plot(xdist,u_data_PDE_StableC[:,ts,0,i,1],label="S-LDO: $s_{l_1} = $" + str(sl) ,linestyle='dashed', color=cp[i])
        
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_ylim([0.1,0.65])
    
    
#%% Compare solutions at certain time-step for different nwidth2

cp = ['blue','orange','green','red','purple','brown']
tsteplist = [100,500,1000]
for ts in tsteplist:
    fig, ax = plt.subplots()
    ax.plot(xdist,u_data_PDE[:,ts],label="Reference",color="black")
    for i in range(len(nwidth1list)):         
        sl = str(2*nwidth1list[i]+1)
        ax.plot(xdist,u_data_PDE_Ridge[:,ts,0,1,i],label="LDO: $s_{l_1} = $" + str(sl) ,linestyle='dashed', color=cp[i])
        # ax.plot(x,u_data_PDE_StableC[:,ts,0,0,i],label="S-LDO: $s_l = $" + str(sl) ,linestyle='solid', color=cp[i])
        
    ax.legend(loc="upper left")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_ylim([0.1,0.65])
    
cp = ['blue','orange','green','red','purple','brown']
tsteplist = [100,500,1000]
for ts in tsteplist:
    fig, ax = plt.subplots()
    ax.plot(xdist,u_data_PDE[:,ts],label="Reference",color="black")
    for i in range(len(nwidth1list)):         
        sl = str(2*nwidth1list[i]+1)
        # ax.plot(x,u_data_PDE_Ridge[:,ts,0,0,i],label="LDO: $s_l = $" + str(sl) ,linestyle='dashed', color=cp[i])
        ax.plot(xdist,u_data_PDE_StableC[:,ts,0,1,i],label="S-LDO: $s_{l_1} = $" + str(sl) ,linestyle='dashed', color=cp[i])
        
    ax.legend()
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u$")
    ax.set_ylim([0.1,0.65])
    
    
#%% Compare errors in time
tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
normval = np.linalg.norm(u_ref,axis=0)
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    errorRidge = np.linalg.norm(u_data_PDE_Ridge[:,:,0,i,1] - u_ref,axis=0)
    errorStableC = np.linalg.norm(u_data_PDE_StableC[:,:,0,i,1] - u_ref,axis=0)
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorRidge/normval,label="LDO: $s_{l_1} = $" + str(sl) ,linestyle='dashed', color=cp[i])

# ax.axvspan(0.0, deltaT*nsteps, alpha=0.2, color='red')
ax.legend(loc="upper right")
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([1e-4,1])
ax.set_yscale("log")

tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    errorRidge = np.linalg.norm(u_data_PDE_Ridge[:,:,0,i,1] - u_ref,axis=0)
    errorStableC = np.linalg.norm(u_data_PDE_StableC[:,:,0,i,1] - u_ref,axis=0)
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorStableC/normval,label="S-LDO: $s_{l_1} = $" + str(sl) ,linestyle='dashed', color=cp[i])

# ax.axvspan(0.0, deltaT*nsteps, alpha=0.2, color='red')
ax.legend()
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([1e-4,1])
ax.set_yscale("log")
    

#%% Error matrix plot
import matplotlib.ticker as ticker 

u_ref = u_data_PDE
normval = np.linalg.norm(u_ref,axis=(0,1))
errorRidge = np.zeros([len(nwidth1list),len(nwidth2list)])
errorStableC = np.zeros([len(nwidth1list),len(nwidth2list)])                      
for i in range(len(nwidth1list)):  
    for j in range(len(nwidth2list)):
        errorRidge[i,j] = np.linalg.norm(u_data_PDE_Ridge[:,:,0,i,j] - u_ref,axis=(0,1))/normval
        errorStableC[i,j] = np.linalg.norm(u_data_PDE_StableC[:,:,0,i,j] - u_ref,axis=(0,1))/normval
        
from matplotlib.colors import LogNorm        
vmin = 5e-3
vmax = 1e-1        
# labellist = np.array([None,3,None,5,None,7,None,11])
labellist = np.append([0],2*np.array(nwidth1list)+1)
fig, ax = plt.subplots()
cs = ax.imshow(errorRidge, norm=LogNorm(vmin=vmin, vmax=vmax))
ax.set_ylabel(r"$s_{l_1}$")        
ax.set_xlabel(r"$s_{l_2}$")
ax.set_yticklabels(labellist)
ax.set_xticklabels(labellist)
space = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(space))   
ax.yaxis.set_major_locator(ticker.MultipleLocator(space))   
cbar = plt.colorbar(cs)
cbar.ax.set_xlabel(r"$\epsilon_{xt}$")

fig, ax = plt.subplots()
cs = ax.imshow(errorStableC, norm=LogNorm(vmin=vmin, vmax=vmax))
ax.set_ylabel(r"$s_{l_1}$")        
ax.set_xlabel(r"$s_{l_2}$")    
ax.set_yticklabels(labellist)
ax.set_xticklabels(labellist)
space = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(space))   
ax.yaxis.set_major_locator(ticker.MultipleLocator(space))   
cbar = plt.colorbar(cs)
cbar.ax.set_xlabel(r"$\epsilon_{xt}$")

#%% Solve Burgers based on computed derivatives
iplot = 0
isave = 0
params = {}
params['learntype'] = "Ridge"   ## Ridge, Lasso, Elastic-Net, Stable-C
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['epsConst1'] = 0
params['epsConst2'] = 0
params['ieqtype'] = 1
params['iBC'] = iBC
nwidth1 = 2    ##[1,2,3,5,10,20], [1]
nwidth2 =  2   ##[2,3,5,10,20,40], [2]
reglist1 = [10,1,1e-1,1e-2,1e-3]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
reglist2 = [10,1,1e-1,1e-2,1e-3]                ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]
outpath = "Figures/"
countlda = -1

u_data_PDE_Ridge_reg = np.zeros([N,nsteps2,len(reglist1),len(reglist2)])

countlda1 = -1
for lda1 in reglist1:
    countlda1 = countlda1 + 1
    countlda2 = -1
    for lda2 in reglist2:
        countlda2 = countlda2 + 1
        params['lda1'] = lda1
        params['lda2'] = lda2
        params['lda'] = lda1
        params['nwidthAdvec'] = nwidth1
        params['nwidthDiff'] = nwidth2
        output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
        advecOpt = output['advecOpt']
        diffOpt = output['diffOpt']
        u_data_PDE_Ridge_reg[:,:,countlda1,countlda2] = solvePDE_Opt(paramsODE_opt,advecOpt,diffOpt)

#%% Error matrix plot

fig, ax = plt.subplots()
normval = np.linalg.norm(u_ref,axis=(0,1))
errorRidge_Reg = np.zeros([len(reglist1),len(reglist2)])
for i in range(len(reglist1)):  
    for j in range(len(reglist2)):
        errorRidge_Reg[i,j] = np.linalg.norm(u_data_PDE_Ridge_reg[:,:,i,j] - u_ref,axis=(0,1))/normval
        
from matplotlib.colors import LogNorm        
vmin = 5e-3
vmax = 1e-1        
labellist_y = np.append(0,np.array(reglist1))
labellist_x = np.append(0,np.array(reglist2))
fig, ax = plt.subplots()
cs = ax.imshow(errorRidge_Reg, norm=LogNorm(vmin=vmin, vmax=vmax))
ax.set_ylabel(r"$\beta_1$")        
ax.set_xlabel(r"$\beta_2$")
ax.set_yticklabels(labellist_y)
ax.set_xticklabels(labellist_x)  
cbar = plt.colorbar(cs)
cbar.ax.set_xlabel(r"$\epsilon_{xt}$")
