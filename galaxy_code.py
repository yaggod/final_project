import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


ae = 149597870700
parcek = ae*206265
inner_eccentricity=0.3
outer_eccentricity=0.9
angular_offset=0.0004/ae # от 0.0003/масштаб до 0.0007/масштаб
core_radius=1000*ae
galaxy_radius=20000*ae
vxc=0
vyc=0
xc=0
yc=0
N = 5000


"""
Функция распределения частиц в галактике
inner_eccentricity - Эксцентричность внутреннего эллипса
outer_eccentricity - Эксцентриситет внешнего эллипса
angular_offset - Угловое смещение на парсек
core_radius - Внутренний радиус ядра
galaxy_radius - Радиус галактики
N - Количество звезд
vxc - "х" компонента скорости центра
vyc - "у" компонента скорости центра
xc - начальные координаты центра
yc - начальные координаты центра
"""

distant_radius = galaxy_radius * 2 # Радиус, после которого все волны
                                   # плотности должны иметь округлую форму.
# Создания массивов данных для частиц
theta = np.ndarray(shape=(N))
angle = np.ndarray(shape=(N))
m_a = np.ndarray(shape=(N))
m_b = np.ndarray(shape=(N))
coordinate = np.ndarray(shape=(N, 2))
velocity = np.ndarray(shape=(N, 2))

# Функция рассчитывает эксцентриситет
def eccentricity(r):

    if r < core_radius:
        return 1 + (r / core_radius) * (inner_eccentricity-1)

    elif r > core_radius and r <= galaxy_radius:
        a = galaxy_radius - core_radius
        b = outer_eccentricity - inner_eccentricity
        return inner_eccentricity + (r - core_radius) / a * b

    elif r > galaxy_radius and r < distant_radius:
        a = distant_radius - galaxy_radius
        b = 1 - outer_eccentricity
        return outer_eccentricity + (r - galaxy_radius) / a * b

    else:
        return 1

# Инициализация  звёзд
X = np.random.uniform(-galaxy_radius, galaxy_radius, N)
Y = np.random.uniform(-galaxy_radius, galaxy_radius, N)
R = np.sqrt(X*X+Y*Y)
m_a = R + 1000
angle = R * angular_offset
theta = np.random.uniform(0, 360, N)
m_b = np.ndarray(shape=(N))
for i in range(N):
    m_b[i] = R[i] * eccentricity(R[i])

# Анимирование
fig, ax  = plt.subplots(figsize=(10,10))
ax.set_xlim(-50000*ae, 50000*ae)
ax.set_ylim(-50000*ae, 50000*ae)
ax.set_facecolor('black')
stars, = plt.plot([],[],'.', ms='1', color='white')

coor_x = np.ndarray(shape=(N))
coor_y = np.ndarray(shape=(N))

def func(timestep=1):
    for i in range(0, N, 1):
        theta[i] += 0.05 * timestep
        alpha = np.ndarray(shape=(N))
        alpha[i] = theta[i] * np.pi / 180.0
        x = xc + m_a[i]*np.cos(alpha[i])*np.cos(-angle[i]) - m_b[i]*np.sin(alpha[i])*np.sin(-angle[i])
        y = yc + m_a[i]*np.cos(alpha[i])*np.sin(-angle[i]) + m_b[i]*np.sin(alpha[i])*np.cos(-angle[i])
        coor_x[i] = x
        coor_y[i] = y
    return coor_x, coor_y


def update(j):
    stars.set_data(func(timestep=j)[0],func(timestep=j)[1])

ani = animation.FuncAnimation(fig,
                              update,
                              frames=300,
                              interval=30)


# ani.save('7.gif',fps=30)
plt.show()
