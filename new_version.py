import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint	
def collision(x1,vx1,radius,mass1):
    r12=np.sqrt((x1-xzc)**2)
    if y1>=0:
        fet=np.arccos(x1/r12)
    elif y1<0:
        2*np.pi-np.arccos(x1/r12)
    if r12<= radius:
        VX1=((v1*np.cos(teta-fet)*(m1))/(m1))*np.cos(fet)+v1*np.sin(teta-fet)*np.cos(fet+np.pi/2)
        VY1=((v1*np.cos(teta-fet)*(m1))*np.sin(fet)+v1*np.sin(teta-fet)*np.sin(fet+np.pi/2))
    if VY1>=0:
        teta=np.arccos(VX1/v1)
    elif VY1<0:
        teta=2*np.pi-np.arccos(VX1/v1)
    else:
      VX1=vx1
    return VX1
def move_func(s,t):
    x1,v_x1=s
    dx1dt=v_x1
    dv_x1dt=0
    return dx1dt,dv_x1dt

K=0
T=5
n=200
mass1=3
radius=5.5
xzc=2
y10=0
x10=-20
v10=10*np.cos(np.pi/6)
x1=[x10]
y1=[y10]
tau=np.linspace(0,T,n)
for k in range (n-1):
    t=[tau[k],tau[k+1]]
    s0=x10,v10
    sol=odeint(move_func,s0,t)
    x10=sol[1,0]
    x1.append(x10)
    V10=sol[1,1]
    res=collision(x10,v10,radius,mass1)
    v10=res
fig, ax=plt.subplots()
ball_1,=plt.plot([],[],'o',color='r',ms=25)
plt.plot([2,2],[-1,5],ms='15')
def animate(i):
    ball_1.set_data((x1 [i],0))
ani= FuncAnimation(fig,animate,frames=n,interval=30)
ax.set_xlim(-50,100)
ax.set_ylim(-1,1)
plt.show()