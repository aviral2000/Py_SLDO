import numpy as np    
import matplotlib.pyplot as plt
from petsc4py import PETSc
import math
from scipy.optimize import minimize, Bounds
from petsc4py import PETSc
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'large',
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'large',
         'ytick.labelsize':'large'}
pylab.rcParams.update(params)

#%% Run 2D advection problem: Modified from the initial version in https://scipython.com/book/chapter-7-matplotlib/problems/p72/the-two-dimensional-advection-equation/
# Domain size
w = h = 10.
# Intervals in x-, y- directions
dx = dy = 0.1
nx, ny = int(w/dx), int(h/dy)
# Time interval
dt = 0.05

# The discretized function on the domain. Note that rows (first index)
# correspond to the y-axis and columns (second index) to the x-index
u0 = np.zeros((ny, nx))
u = np.zeros((ny, nx))
x, y =  np.meshgrid(np.arange(0,nx*dx,dx), np.arange(0,ny*dy,dy))
# Initial state: a two-dimensional Gaussian function
# cx, cy, alpha = 5, 5, 2
cx, cy, alpha = 2, 5, 1
uinit = np.exp(-((x-cx)**2+(y-cy)**2)/alpha**2)

## Modified setup
vfacx = 0.5
vfacy = 0.5
# Number of timesteps:
nsteps = 500

paramsPDE = {}
paramsPDE['nsteps'] = nsteps
paramsPDE['vfacx'] = 0.5
paramsPDE['vfacy'] = 0.5
paramsPDE['dt'] = dt
paramsPDE['nx'] = nx
paramsPDE['ny'] = ny
paramsPDE['uinit'] = uinit

def solvePDE(params):
    nsteps = params['nsteps']
    vfacx = params['vfacx']
    vfacy = params['vfacy']
    dt = params['dt']
    nx = params['nx']
    ny = params['ny']
    uinit = params['uinit']
    u_data = np.zeros([ny,nx,nsteps])
    ut_data = np.zeros([ny,nx,nsteps])
    uold = uinit
    for m in range(nsteps):
        indx = np.arange(nx)
        indy = np.arange(ny)
        indxm1 = indx-1
        indym1 = indy-1    
        indxp1 = indx+1
        indyp1 = indy+1    
        indxp1[nx-1] = 0
        indyp1[ny-1] = 0
    
        deltau_x = (dt*vfacx/dx)*(uold[:,indx] - uold[:,indxm1])
        deltau_y = (dt*vfacy/dy)*(uold[indy,:] - uold[indym1,:])
        # deltau_y = (0.5*dt*vfacy/dy)*(uold[indyp1,:] - uold[indym1,:])
        u = uold - (deltau_x + deltau_y)
        u_data[:,:,m] = u
        ut_data[:,:,m] = (u-uold)/dt
        uold = u
    
    return(u_data,ut_data)

[u_data,ut_data] = solvePDE(paramsPDE)


#%% Animate the results

fig, ax = plt.subplots(1, 1, figsize = (6, 6))

def animate(i):
    ax.cla() # clear the previous image
    # ax.plot(xdist[:], u_data[:,i],label="True") # plot the line
    # ax.set_ylim([-1.0, 1.0]) # fix the y axis
    ax.imshow(u_data[:,:,i])
    ax.invert_yaxis()
    
from matplotlib import animation
anim = animation.FuncAnimation(fig, animate, frames = range(0,nsteps,20), interval = 1, 
                               blit = False)

plt.show()

#%% Set BC index function for regression
def setBCindex(iBC, kx, ky, nwidthx, nwidthy, nx, ny):
    if(iBC == 1):
        if(kx<nwidthx):
            inda = range(0,kx+nwidthx+1)
            indb = range(nx-(nwidthx-kx),nx)
            indx = list(indb) + list(inda)      
        elif(kx>nx-(nwidthx+1)):
            inda = range(kx-nwidthx,nx)
            indb = range(0,(kx+nwidthx+1)-nx)
            indx = list(inda) + list(indb)
        else:
            indx = range(kx-nwidthx,kx+nwidthx+1)
            
        if(ky<nwidthy):
            inda = range(0,ky+nwidthy+1)
            indb = range(ny-(nwidthy-ky),ny)
            indy = list(indb) + list(inda)      
        elif(ky>ny-(nwidthy+1)):
            inda = range(ky-nwidthy,ny)
            indb = range(0,(ky+nwidthy+1)-ny)
            indy = list(inda) + list(indb)
        else:
            indy = range(ky-nwidthy,ky+nwidthy+1)
    elif(iBC == 0):
        print("Does not work yet!")
        if(kx<nwidthx):
            inda = range(0,kx+nwidthx+1)
            indx = list(inda)      
        elif(kx>nx-(nwidthx+1)):
            inda = range(kx-nwidthx,nx)
            indx = list(inda)
        else:
            indx = range(kx-nwidthx,kx+nwidthx+1)
            
        if(ky<nwidthy):
            inda = range(0,ky+nwidthy+1)
            indy = list(inda)      
        elif(ky>ny-(nwidthy+1)):
            inda = range(ky-nwidthy,ny)
            indy = list(inda)
        else:
            indy = range(ky-nwidthy,ky+nwidthy+1)            
            
    return(indx,indy)

#%% Function for learning the operator

def LearnDifferentialOperator(params,iplot,isave,u_data,ut_data):
    iBC = params['iBC']
    nwidthx = params['nwidthx']
    nwidthy = params['nwidthy']
    learntype = params['learntype']
    l1_ratio = params['l1_ratio']
    lda = params['lda']
    nsteps = params['nsteps']
    nx = params['nx']
    ny = params['ny']
    vfacx = params['vfacx']
    vfacy = params['vfacy']
    ieqtype = params['ieqtype']
    
    npts = ny*nx
    totalwidthx = (2*nwidthx)+1
    totalwidthy = (2*nwidthy)+1
    totalwidth = totalwidthx + totalwidthy
    connectMatx = np.zeros([npts,totalwidthx], dtype=int)
    connectMaty = np.zeros([npts,totalwidthy], dtype=int)
    u_data_vec = np.zeros([npts,nsteps])
    for ky in range(0,ny):
        for kx in range(0,nx):
            ipt = kx + (ky*nx)
            [indx,indy] = setBCindex(iBC, kx, ky, nwidthx, nwidthy, nx, ny) 
            iptlist = np.zeros([totalwidthy], dtype=int)
            count = 0
            for jy in indy:
                ipt2 = kx + (jy*nx)
                iptlist[count] = int(ipt2)
                count = count + 1            
            connectMaty[ipt,:] = iptlist
            
            iptlist = np.zeros([totalwidthx], dtype=int)
            count = 0
            for jx in indx:
                ipt2 = jx + (ky*nx)
                iptlist[count] = int(ipt2)
                count = count + 1            
            connectMatx[ipt,:] = iptlist
            
            u_data_vec[ipt,:] = u_data[ky,kx,:]
        print("Completed, ky:", ky)
            
    advecOptx = np.zeros([npts,totalwidthx])
    advecOpty = np.zeros([npts,totalwidthy])
    x0 = np.ones([totalwidth])        
    for ky in range(0,ny):
        for kx in range(0,nx):
            bvec = -ut_data[ky,kx,:]
            Amat = np.zeros([nsteps,totalwidth])
            ipt = kx + (ky*nx)
            for i in range(nsteps):
                Amat[i,0:totalwidthx] = vfacx*u_data_vec[connectMatx[ipt],i]
                Amat[i,totalwidthx:totalwidth] = vfacy*u_data_vec[connectMaty[ipt],i]
    
            if(learntype == "Ridge"):            
                LHS = np.matmul(Amat.T,Amat) + lda*np.eye(totalwidth)
                RHS = np.matmul(Amat.T,bvec)
                coefftmp = np.linalg.solve(LHS,RHS)
                advecOptx[ipt,:] = coefftmp[0:totalwidthx]
                advecOpty[ipt,:] = coefftmp[totalwidthx:totalwidth]    
            elif(learntype == "Stable-C"):
                def funcTomin(x,Amat,bvec):
                    return(np.sum(np.square(np.matmul(Amat,x) - bvec)))
        
                def funcTomin_der(x,Amat,bvec):
                    tmp = np.matmul(Amat,x) - bvec
                    der = 2*np.matmul(Amat.T,tmp)
                    return(der)
                
                twx = totalwidthx+nwidthy
                if(ieqtype == 1):
                    ineq_cons = {'type': 'ineq',
                             'fun' : lambda x: lda*np.array([x[nwidthx] - np.sum(np.abs(x[0:nwidthx])) - 
                                                         np.sum(np.abs(x[nwidthx+1:totalwidthx])),
                                                         x[twx] - np.sum(np.abs(x[totalwidthx:twx])) - 
                                                         np.sum(np.abs(x[twx+1:totalwidth]))]),
                             'jac' : lambda x: lda*np.array([np.concatenate([-1*np.sign(x[0:nwidthx]),[1],
                                                                        -1*np.sign(x[nwidthx+1:totalwidthx]),np.zeros(totalwidthy)]),
                                                             np.concatenate([np.zeros(totalwidthx),-1*np.sign(x[totalwidthx:twx]),[1],
                                                                        -1*np.sign(x[twx+1:totalwidth])])])}
                elif(ieqtype == 2):
                    ineq_cons = {'type': 'ineq',
                             'fun' : lambda x: lda*np.array([vfacx*(x[nwidthx] - np.sum(np.abs(x[0:nwidthx])) - 
                                                         np.sum(np.abs(x[nwidthx+1:totalwidthx]))) + 
                                                         vfacy*(x[twx] - np.sum(np.abs(x[totalwidthx:twx])) - 
                                                         np.sum(np.abs(x[twx+1:totalwidth])))]),
                             'jac' : lambda x: lda*np.array([np.concatenate([-vfacx*np.sign(x[0:nwidthx]),[vfacx],
                                                                        -vfacx*np.sign(x[nwidthx+1:totalwidthx]),
                                                             -vfacy*np.sign(x[totalwidthx:twx]),[vfacy],
                                                                        -vfacy*np.sign(x[twx+1:totalwidth])])])}                    
                
                res = minimize(funcTomin, x0, args=(Amat,bvec), method='SLSQP', jac=funcTomin_der,
                               constraints=[ineq_cons], options={'ftol': 1e-6, 'disp':True, 'maxiter': 10000})
                advecOptx[ipt,:] = res.x[0:totalwidthx]
                advecOpty[ipt,:] = res.x[totalwidthx:totalwidth]
                x0 = res.x
                                                    
            print("Done, iteration = ", kx, ky)        
        
    output={}    
    output['advecOptx'] = advecOptx
    output['advecOpty'] = advecOpty
    output['connectMatx'] = connectMatx
    output['connectMaty'] = connectMaty
    return(output)

#%% Generate matrix with list of nearby nodes and construct advecOptx/y for stability analysis

params = {}
params['iBC'] = 1
params['nwidthx'] = 2
params['nwidthy'] = 2
params['learntype'] = "Stable-C"
params['ieqtype'] = 1
params['l1_ratio'] = 0.5
params['lda'] = 1.0
params['nsteps'] = nsteps
params['nx'] = nx
params['ny'] = ny
params['vfacx'] = vfacx
params['vfacy'] = vfacy

iplot = 0
isave = 0
output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
advecOptx = output['advecOptx']
advecOpty = output['advecOpty']
connectMatx = output['connectMatx']
connectMaty = output['connectMaty']

npts = ny*nx
advecMatx = np.zeros([npts,npts])
advecMaty = np.zeros([npts,npts])
for ky in range(0,ny):
    for kx in range(0,nx):
        ipt = kx + (ky*nx)
        advecMatx[ipt,connectMatx[ipt]] = advecOptx[ipt,:]  
        advecMaty[ipt,connectMaty[ipt]] = advecOpty[ipt,:]                        
        
#%% Linear stability analysis

# [lda,vec] = np.linalg.eig(-advecMatx)

# fig, ax = plt.subplots()  
# ax.scatter(np.real(lda), np.imag(lda), marker=".")
# ax.axvspan(-10, 0.0, alpha=0.2, color='red')
# ax.legend(loc="upper right")
# # ax.set_ylim([-8,8])
# # ax.set_xlim([-4,4])

# [lda,vec] = np.linalg.eig(-advecMaty)

# fig, ax = plt.subplots()  
# ax.scatter(np.real(lda), np.imag(lda), marker=".")
# ax.axvspan(-10, 0.0, alpha=0.2, color='red')
# ax.legend(loc="upper right")
# # ax.set_ylim([-8,8])
# # ax.set_xlim([-4,4])
            


#%% Use computed operators to solve the equation 

fac = 1
dt_new = dt/fac
nsteps2 = nsteps*fac

paramsPDE = {}
paramsPDE['nsteps'] = nsteps2
paramsPDE['vfacx'] = 0.5
paramsPDE['vfacy'] = 0.5
paramsPDE['dt'] = dt_new
paramsPDE['nx'] = nx
paramsPDE['ny'] = ny
paramsPDE['uinit'] = uinit

paramsPDE_Opt = {}
paramsPDE_Opt['dt'] = dt_new
paramsPDE_Opt['nsteps'] = nsteps2
paramsPDE_Opt['uinit'] = uinit
paramsPDE_Opt['vfacx'] = vfacx
paramsPDE_Opt['vfacy'] = vfacy
paramsPDE_Opt['npts'] = npts

def solvePDE_Opt(paramsPDE_Opt,advecOptx,advecOpty,connectMatx,connectMaty):
    dt_new = paramsPDE_Opt['dt']
    nsteps2 = paramsPDE_Opt['nsteps']
    uinit = paramsPDE_Opt['uinit']
    vfacx = paramsPDE_Opt['vfacx']
    vfacy = paramsPDE_Opt['vfacy']
    npts = paramsPDE_Opt['npts']
    
    u_new =  np.zeros([npts,nsteps2])
    uold = np.reshape(uinit,npts)
    deltaux = np.zeros([npts])
    deltauy = np.zeros([npts])
    for it in range(nsteps2):
        for i in range(npts):
            deltaux[i] = -vfacx*np.inner(advecOptx[i,:],uold[connectMatx[i]])
            deltauy[i] = -vfacy*np.inner(advecOpty[i,:],uold[connectMaty[i]])
        # deltau1 = -vfac*np.matmul(advecMat,uold)
        # print(deltau1)
        u_new[:,it] = uold + dt_new * (deltaux + deltauy)
        uold = u_new[:,it]
        if(it%100 == 0):
            print("Iteration completed: ", it)
            
    return(u_new)


iplot = 0
isave = 0
params = {}
params['iBC'] = 1
params['learntype'] = "Ridge"
params['ieqtype'] = 2
params['l1_ratio'] = 0.5
params['nsteps'] = nsteps
params['nx'] = nx
params['ny'] = ny
params['vfacx'] = vfacx
params['vfacy'] = vfacy
nwidth1list = [1,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list = [1,3,5,10,20]   ##[2,3,5,10,20,40], [2]
reglist = [1e-2]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]

[u_data_PDE, tmp] = solvePDE(paramsPDE)

u_data_PDE_Ridge = np.zeros([npts,nsteps2,len(reglist),len(nwidth1list),len(nwidth2list)])

countlda = -1
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthx'] = nwidth1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthy'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            advecOptx = output['advecOptx']
            advecOpty = output['advecOpty']
            connectMatx = output['connectMatx']
            connectMaty = output['connectMaty']
            u_data_PDE_Ridge[:,:,countlda,countnw1,countnw2] = solvePDE_Opt(paramsPDE_Opt,advecOptx,
                                                                            advecOpty,connectMatx,connectMaty)
            

#%% Do the same for Stable-C
iplot = 0
isave = 0
params = {}
params['iBC'] = 1
params['learntype'] = "Stable-C"
params['ieqtype'] = 2
params['l1_ratio'] = 0.5
params['nsteps'] = nsteps
params['nx'] = nx
params['ny'] = ny
params['vfacx'] = vfacx
params['vfacy'] = vfacy
nwidth1list = [1,3,5,10,20]    ##[1,2,3,5,10,20], [1]
nwidth2list = [1,3,5,10,20]   ##[2,3,5,10,20,40], [2]
reglist = [1]                  ##[1,1e-1,1e-2,1e-3,1e-4,1e-5,1e-6], [1]

u_data_PDE_StableC = np.zeros([npts,nsteps2,len(reglist),len(nwidth1list),len(nwidth2list)])

countlda = -1
for lda in reglist:
    countlda = countlda + 1
    countnw1 = -1
    params['lda'] = lda
    for nwidth1 in nwidth1list:
        countnw1 = countnw1 + 1
        countnw2 = -1
        params['nwidthx'] = nwidth1
        for nwidth2 in nwidth2list:   
            countnw2 = countnw2 + 1
            params['nwidthy'] = nwidth2
            output = LearnDifferentialOperator(params,iplot,isave,u_data,ut_data)
            advecOptx = output['advecOptx']
            advecOpty = output['advecOpty']
            connectMatx = output['connectMatx']
            connectMaty = output['connectMaty']
            u_data_PDE_StableC[:,:,countlda,countnw1,countnw2] = solvePDE_Opt(paramsPDE_Opt,advecOptx,
                                                                            advecOpty,connectMatx,connectMaty)
        
#%% Animate the results
from matplotlib import animation
u_new_reshape_Ridge = np.reshape(u_data_PDE_Ridge[:,:,0,0,0],[ny,nx,nsteps2])
u_new_reshape_StableC = np.reshape(u_data_PDE_StableC[:,:,0,0,0],[ny,nx,nsteps2])


fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (6, 6))

def animate1(i):
    ax1.cla() # clear the previous image
    # ax.plot(xdist[:], u_data[:,i],label="True") # plot the line
    # ax.set_ylim([-1.0, 1.0]) # fix the y axis
    ax1.imshow(u_data_PDE[:,:,i])
    ax1.invert_yaxis()

ani1 = animation.FuncAnimation(fig, animate1, frames = range(0,nsteps,10), interval = 1, 
                               blit = False)

def animate2(i):
    ax2.cla() # clear the previous image
    # ax.plot(xdist[:], u_data[:,i],label="True") # plot the line
    # ax.set_ylim([-1.0, 1.0]) # fix the y axis
    ax2.imshow(u_new_reshape_Ridge[:,:,i])    
    ax2.invert_yaxis()

ani2 = animation.FuncAnimation(fig, animate2, frames = range(0,nsteps2,10*fac), interval = 1, 
                               blit = False)

def animate3(i):
    ax3.cla() # clear the previous image
    # ax.plot(xdist[:], u_data[:,i],label="True") # plot the line
    # ax.set_ylim([-1.0, 1.0]) # fix the y axis
    ax3.imshow(u_new_reshape_StableC[:,:,i])    
    ax3.invert_yaxis()

ani3 = animation.FuncAnimation(fig, animate3, frames = range(0,nsteps2,10*fac), interval = 1, 
                               blit = False)

plt.show()


#%% Plot solution at different time
import matplotlib.cm as cm

tsteplist = [249]
vmax = 0.8
vmin = 0.0
levels = 11
nticks = 20
labellist_x = np.arange(0,nx*dx,dx)
labellist_y = np.flip(np.arange(0,ny*dy,dy))
labellist_ticks = np.linspace(0, 1.0, levels)

for ts in tsteplist:
    fig, ax = plt.subplots(figsize = (6, 6))
    cs = ax.contourf(x,y,u_data_PDE[:,:,ts], levels=levels, vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar(cs, ticks=np.linspace(0, 1.0, 11))
    # cbar.ax.set_xlabel(r"$u$")
    ax.set_ylabel(r"$y$")        
    ax.set_xlabel(r"$x$")

    fig, ax = plt.subplots(figsize = (6, 6))
    cs = ax.contourf(x,y,u_new_reshape_Ridge[:,:,ts], levels=levels, vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar(cs, ticks=np.linspace(0, 0.8, 11))
    # cbar.set_ticklabels(labellist_ticks)
    ax.set_ylabel(r"$y$")        
    ax.set_xlabel(r"$x$")

        
    fig, ax = plt.subplots(figsize = (6, 6))
    cs = ax.contourf(x,y,u_new_reshape_StableC[:,:,ts], levels=levels, vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar(cs, ticks=np.linspace(0, 1.0, 11))  
    # cbar.set_ticklabels(labellist_ticks)
    ax.set_ylabel(r"$y$")        
    ax.set_xlabel(r"$x$")


#%% Plot error variation in time for stencil size in x
cp = ['blue','orange','green','red','purple','brown']
tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
normval = np.linalg.norm(u_ref,axis=(0,1))
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    u_new_reshape_Ridge = np.reshape(u_data_PDE_Ridge[:,:,0,i,1],[ny,nx,nsteps2])
    errorRidge = np.linalg.norm(u_new_reshape_Ridge - u_ref,axis=(0,1))
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorRidge/normval,label="LDO: $s_{l_x} = $" + str(sl) ,linestyle='dashed', color=cp[i])

# ax.axvspan(0.0, deltaT*nsteps, alpha=0.2, color='red')
ax.legend(loc="lower right")
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([5e-4,1])
ax.set_yscale("log")


tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    u_new_reshape_StableC = np.reshape(u_data_PDE_StableC[:,:,0,i,1],[ny,nx,nsteps2])
    errorStableC = np.linalg.norm(u_new_reshape_StableC - u_ref,axis=(0,1))
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorStableC/normval,label="S-LDO: $s_{l_x} = $" + str(sl) ,linestyle='dashed', color=cp[i])

# ax.axvspan(0.0, deltaT*nsteps, alpha=0.2, color='red')
ax.legend(loc="lower right")
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([5e-4,1])
ax.set_yscale("log")

    
#%% Plot error variation in time for stencil size in y
cp = ['blue','orange','green','red','purple','brown']
tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
normval = np.linalg.norm(u_ref,axis=(0,1))
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    u_new_reshape_Ridge = np.reshape(u_data_PDE_Ridge[:,:,0,1,i],[ny,nx,nsteps2])
    errorRidge = np.linalg.norm(u_new_reshape_Ridge - u_ref,axis=(0,1))
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorRidge/normval,label="LDO: $s_{l_y} = $" + str(sl) ,linestyle='dashed', color=cp[i])

# ax.axvspan(0.0, deltaT*nsteps, alpha=0.2, color='red')
ax.legend(loc="upper right")
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([5e-4,1])
ax.set_yscale("log")


tlist = dt_new*np.arange(nsteps2)
u_ref = u_data_PDE
fig, ax = plt.subplots()
for i in range(len(nwidth1list)):  
    u_new_reshape_StableC = np.reshape(u_data_PDE_StableC[:,:,0,1,i],[ny,nx,nsteps2])
    errorStableC = np.linalg.norm(u_new_reshape_StableC - u_ref,axis=(0,1))
    
    sl = str(2*nwidth1list[i]+1)
    ax.plot(tlist,errorStableC/normval,label="S-LDO: $s_{l_y} = $" + str(sl) ,linestyle='dashed', color=cp[i])

# ax.axvspan(0.0, deltaT*nsteps, alpha=0.2, color='red')
ax.legend()
ax.set_xlabel("$t$")
ax.set_ylabel("$e_u$")
ax.set_xlim([0,dt_new*nsteps2])
ax.set_ylim([5e-4,1])
ax.set_yscale("log")
    
#%% Plot matrix error

import matplotlib.ticker as ticker 

u_ref = u_data_PDE
u_ref_reshape = np.reshape(u_ref,[ny*nx,nsteps2])
normval = np.linalg.norm(u_ref_reshape,axis=(0,1))
errorRidge = np.zeros([len(nwidth1list),len(nwidth2list)])
errorStableC = np.zeros([len(nwidth1list),len(nwidth2list)])                      
for i in range(len(nwidth1list)):  
    for j in range(len(nwidth2list)):
        errorRidge[i,j] = np.linalg.norm(u_data_PDE_Ridge[:,:,0,i,j] - u_ref_reshape, axis=(0,1))/normval
        errorStableC[i,j] = np.linalg.norm(u_data_PDE_StableC[:,:,0,i,j] - u_ref_reshape, axis=(0,1))/normval
        
from matplotlib.colors import LogNorm        
vmin = 5e-2
vmax = 1e-0        
# labellist = np.array([None,3,None,5,None,7,None,11])
labellist = np.append([0],2*np.array(nwidth1list)+1)
fig, ax = plt.subplots()
cs = ax.imshow(errorRidge, norm=LogNorm(vmin=vmin, vmax=vmax))
ax.set_ylabel(r"$s_{l_x}$")        
ax.set_xlabel(r"$s_{l_y}$")
ax.set_yticklabels(labellist)
ax.set_xticklabels(labellist)
space = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(space))   
ax.yaxis.set_major_locator(ticker.MultipleLocator(space))   
cbar = plt.colorbar(cs)
cbar.ax.set_xlabel(r"$\epsilon_{xt}$")

fig, ax = plt.subplots()
cs = ax.imshow(errorStableC, norm=LogNorm(vmin=vmin, vmax=vmax))
ax.set_ylabel(r"$s_{l_x}$")        
ax.set_xlabel(r"$s_{l_y}$")    
ax.set_yticklabels(labellist)
ax.set_xticklabels(labellist)
space = 1
ax.xaxis.set_major_locator(ticker.MultipleLocator(space))   
ax.yaxis.set_major_locator(ticker.MultipleLocator(space))   
cbar = plt.colorbar(cs)
cbar.ax.set_xlabel(r"$\epsilon_{xt}$")