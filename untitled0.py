import numpy as np
from random import uniform
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

def rand(a):
    return uniform(-a, a)

def funcAnimate(frame):

    for i in range(amount):
        Balls[i].set_data([result[frame][i]], [result[frame][i+amount]])


    
            

class ParticleCollection:
    class Particle:
        def __init__(self, x, y, coef):
            self.x = x
            self.y = y
            self.vy = x*coef
            self.vx = -y*coef
      
        def toArray(self):
            return np.array([self.x,self.y,self.vx,self.vy])
  
    def __init__(self, amount):
        self.amount = amount
        self.collection = np.ndarray(shape=(amount), dtype=ParticleCollection.Particle)
  
    def initParticles(self, xLims, yLims, coef, radius, mass):
        self.radius = radius
        self.mass = mass
        for i in range(self.amount):
            self.collection[i] = ParticleCollection.Particle(rand(xLims), 
                                                             rand(yLims), 
                                                             coef)
            #self.collection[i] = ParticleCollection.Particle(5, 10, coef)  

    def solve(self, framecount, alltime):
        result = []
        result.append(self.toCrt())
        t = np.linspace(0,alltime,framecount)
        for i in range(framecount-1):
            self.temp = solve_ivp(ParticleCollection.solveDiff,
                                    (t[i], t[i+1]), 
                                    result[-1],
                                    args=(self.mass, self.radius))["y"].T
            result.append(self.temp[-1])
        return result
        
    def toCrt(self):
        crt = []
        for i in self.collection:
            crt.append(i.toArray())
            
        return np.array(crt).T.flatten();
    
    @staticmethod
    def solveDiff(t, cort, mass, radius):
        amount = int(len(cort)/4)
        x, y, vx, vy = np.split(cort, 4)
        
        # return np.concatenate((vx,vy,vx*0, vy*0))      
        
        dvy_dt = np.zeros(shape=amount)
        dvx_dt = np.zeros(shape=amount)
        for i in range(amount):
            for j in range(i+1, amount):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                zn = (dx*dx + dy*dy)
                if (zn <= (radius*2)):
                    zn = radius*2
                    
                zn = zn**1.5
                ch = 6.67e-11*mass
                Ax = ch*dx/zn
                Ay = ch*dy/zn
                dvx_dt[i] += Ax
                dvy_dt[i] += Ay
                dvx_dt[j] -= Ax
                dvy_dt[j] -= Ay    
        result = np.concatenate((vx, vy, dvx_dt, dvy_dt))
        return result


amount = 100#int(input("Количество тел: "))
framecount = 30 #int(input("Количество кадров: "))
xLim =  8      #float(input("Лимит по оси X: "))
yLim =  4      #float(input("Лимит по оси Y: "))
coef =  0.00001     #float(input("Коэффициент скорости: "))
radius = 0.01     #float(input("Радиусы тел: "))
mass =  200      #float(input("Массы тел: "))
alltime = float(input("Введите все время: "))

A = ParticleCollection(amount)
A.initParticles(xLim, yLim, coef, radius, mass)
result = A.solve(30,alltime)

Balls = np.ndarray(shape=(amount), dtype=Line2D)
fig, ax = plt.subplots()

for i in range(amount):     
    Balls[i], = plt.plot([],[],'o',label='Ball', ms=2)

print("Solved!")


plt.xlim(-xLim*2,xLim*2)
plt.ylim(-yLim*2,yLim*2)

ani = FuncAnimation(fig, funcAnimate, frames=framecount)
ani.save("adfkjfk.gif", writer="pillow")
