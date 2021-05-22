import numpy as np
from random import uniform
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

def rand(a):
    return uniform(-a, a)

def funcAnimate(frame):
    print(f"\rDrawing frame {frame}")
    amount = len(count[frame])
    for i in range(amount):
         Balls[i].set_data([result[frame][i]], [result[frame][i+amount]])
         Balls[i].set_ms(2*ParticleCollection.calculateRC(count[frame][i]))
    for i in range(amount,len(count[0])):
        Balls[i].set_data([result[frame][0]], [result[frame][amount]])


    
            

class ParticleCollection:
    class Particle:
        def __init__(self, x, y, coef):
            self.x = x
            self.y = y
            self.vy = x*coef
            self.vx = -y*coef
      
        def toArray(self):
            return np.array([self.x,self.y,self.vx,self.vy])
  
    def __init__(self):
        print()
  
    def preInitParticles(self,radius,mass,coef):
        self.radius = radius
        self.mass = mass
        self.coef = coef
        self.x = 0
        
    def initParticles(self, xLims):
        
        for i in range(self.amount):
            self.collection[i] = ParticleCollection.Particle(rand(xLims), 
                                                             rand(xLims), 
                                                             self.coef)
    def setCollection(self, coll):
        self.collection = coll
        self.amount = len(coll)
    
    def getCircle(self,amount, frm, to):
        if (to >= self.x): self.x = to
        temp = np.ndarray(shape=(amount), dtype=ParticleCollection.Particle)
        for i in range(amount):
            alpha = rand(np.pi)
            # alpha = np.linspace(0,2*np.pi,amount)
            x = uniform(frm, to)
            temp[i] = ParticleCollection.Particle(x*np.cos(alpha),
                                                  x*np.sin(alpha),
                                                  self.coef)
        return temp

    def solve(self, framecount, alltime):
        counter = np.array([1]*self.amount)
        self.radius = self.x/self.radius
        result = []
        count = []
        res, cnt = self.combineObjects(self.toCrt(), counter)
        result.append(res)
        count.append(cnt)
        t = np.linspace(0,alltime,framecount)     
        for i in range(framecount-1):
            print("\rframe:" , i,end="")
            # print(f"m0: {self.mass*count[0]}")
            # print(f"m1: {self.mass*count[0]}")
            # print(f"m2: {self.mass*count[0]}")
            self.temp = solve_ivp(ParticleCollection.solveDiff,
                                    (t[i], t[i+1]), 
                                    result[-1],
                                    args=(self.mass, self.radius, count[-1]))["y"].T
            
            res, cnt = self.combineObjects(self.temp[-1], count[-1])
            result.append(res)
            count.append(cnt)
            

        
        return result, count
        
    def toCrt(self):
        crt = []
        for i in self.collection:
            crt.append(i.toArray())
            
        return np.array(crt).T.flatten();
   
    
    def combineObjects(self, crt, amounts):
        
        x,y,vx,vy = np.split(crt, 4);
        
        isntCalculated = True
        while (isntCalculated):
            isntCalculated = False
            amount = len(x)
            # print("Length:", len(crt))
            # print("Amount:", amount)
            for i in range(amount):
                
                for j in range(i+1,amount):
                    far = ParticleCollection.howFar(x[i], x[j], y[i], y[j])
                    allowedDist = self.radius* \
                        (ParticleCollection.calculateRC(amounts[i]) + \
                         ParticleCollection.calculateRC(amounts[j]))
                    if (far < allowedDist):
                        # print(f"trueDist: {far}, allowedDist: {allowedDist}")
                        isntCalculated = True
                        amounts[i] += amounts[j]
                        vx[i] = ParticleCollection.getImpulse(amounts[i], 
                                                              amounts[j], 
                                                              vx[i], vx[j])
                        vy[i] =ParticleCollection. getImpulse(amounts[i], 
                                                              amounts[j], 
                                                              vy[i], vy[j])
                        x = np.delete(x, j)
                        y = np.delete(y, j)
                        vx = np.delete(vx, j)
                        vy = np.delete(vy, j)
                        amounts = np.delete(amounts, j)
                        break;
                if (isntCalculated):
                    break    
        return np.concatenate((x,y,vx,vy)), amounts
                
    
    
    @staticmethod
    def getImpulse(c1, c2, v1, v2):
        return (c1*v1 + c2*v2)/(c1+c2)
    
    @staticmethod
    def calculateRC(amount):
        # print("Amount:", amount)
        return amount**(1/3)
    
    @staticmethod
    def howFar(x1,x2,y1,y2):
        return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    
    @staticmethod
    def solveDiff(t, cort, mass, radius, count):
        
        amount = int(len(cort)/4)
        x, y, vx, vy = np.split(cort, 4)
        
        
        
        # return np.concatenate((vx,vy,vx*0, vy*0))      
        
        dvy_dt = np.zeros(shape=amount)
        dvx_dt = np.zeros(shape=amount)
        for i in range(amount):
            for j in range(i+1, amount):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                # print(dx, " ", dy)
                zn = (dx*dx + dy*dy)
                if (zn <= (radius*radius*4)):
                    zn = radius*radius*4
                  
                    #ЗВТ
                zn = zn**1.5
                ch = 6.67e-11*mass
                Ax = ch*dx/zn
                Ay = ch*dy/zn
                dvx_dt[i] += Ax*count[j]
                dvy_dt[i] += Ay*count[j]
                dvx_dt[j] -= Ax*count[i]
                dvy_dt[j] -= Ay*count[i]
                
                
        result = np.concatenate((vx, vy, dvx_dt, dvy_dt))
        return result


framecount = int(input("Количество кадров: "))
coef = float(input("Коэффициент скорости: "))
radiusCoef =float(input("Во сколько раз радиус тел меньше максимальных координат: "))
mass = float(input("Массы начальных частиц: "))
alltime = float(input("Введите все время: "))


A = ParticleCollection()
A.preInitParticles(radiusCoef, mass, coef)
# p1 = A.getCircle(30,0,7000)
# p2 = A.getCircle(55,7e3,5e6)
# p3 = A.getCircle(60,5000000,6000000)

a = []
circleCount = int(input("Сколько \"колец\" задействовано в модели: "))
for i in range(circleCount):
    print(f"\"Кольцо №{i}\":")
    am = int(input("Количество частиц"))
    print("Расстояние до центра:")
    mn = float(input("Минимальное: "))
    mx = float(input("Максимальное: "))
    a.append(A.getCircle(am,mn,mx))
    



allP = np.array(a).flatten()

# allP = A.getCircle(20,70,100)

A.setCollection(allP)

amount = len(allP)

result,count = A.solve(framecount, alltime)



fig, ax = plt.subplots()
Balls = np.ndarray(shape=(len(count[0])), dtype=Line2D)

for i in range(len(count[0])):
    Balls[i], = plt.plot([],[],'o',label='Ball', color='green')

print("Solved!")

plt.xlim(-A.x*2,A.x*2)
plt.ylim(-A.x*2,A.x*2)


ani = FuncAnimation(fig, funcAnimate, frames=framecount, interval = 100)
ani.save("adfkjfk.gif", writer="pillow")
