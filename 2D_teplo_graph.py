import numpy as np
import matplotlib.pyplot as plt




def q(x, y,t):
    return x+x**2-5*y-2*t


def resh(x,y,t):
    return t*x**2+x*t-2+y**3+y*t


# Начальное условие
def phi(x, y):
    return y**3-2


# теплоизолированные стенки
# ГУ для оси x
def alpha_x():
    alpha_x1 = 1
    alpha_x2 = 1

    return [alpha_x1, alpha_x2]


def beta_x():
    beta_x1 = 0
    beta_x2 = 0

    return [beta_x1, beta_x2]


def gamma_x(y, t):
    gamma_x1 = 3*t
    gamma_x2 = -t

    return [gamma_x1, gamma_x2]


# ГУ для оси y
def alpha_y():
    alpha_y1 = 1
    alpha_y2 = 1

    return [alpha_y1, alpha_y2]


def beta_y():
    beta_y1 = 0
    beta_y2 = 0

    return [beta_y1, beta_y2]


def gamma_y(x, t):
    gamma_y1 = 12+t
    gamma_y2 = 12+t
    return [gamma_y1, gamma_y2]


# условие устойчивости Куранта
def Kurant_condition(h_x, h_y, a):
    t_x = h_x ** 2 / 2 / a ** 2 / 2
    t_y = h_y ** 2 / 2 / a ** 2 / 2

    t = min(t_x, t_y)

    return t

def solution(x, y, h_x, h_y, Nt, a = 1, r = 0.25):
    t = Kurant_condition(h_x, h_y, a)
    c_x = a ** 2 * t / h_x ** 2
    c_y = a ** 2 * t / h_y ** 2
    u = []
    y_x_layer = np.zeros((len(y), len(x)))
    for j in range(len(y)):
        for i in range(len(x)):
            y_x_layer[j][i] = y[j]**3-2
    t_prev_layer = y_x_layer
    u.append(y_x_layer)
    q_ = np.zeros((len(y), len(x)))
    for j in range(1, len(y) - 1):
        for i in range(1, len(x) - 1):
            q_[j][i] = q(x[i],y[j],Nt*t)
    for t_i in range(Nt):
        y_x_layer = np.zeros((len(y), len(x)))
        for j in range(1,len(y)-1):
            for i in range(1,len(x)-1):
                y_x_layer[j][i] = (c_x * (t_prev_layer[j][i + 1] - 2 * t_prev_layer[j][i] + t_prev_layer[j][i - 1]) +
                                   c_y * (t_prev_layer[j + 1][i] - 2 * t_prev_layer[j][i] + t_prev_layer[j - 1][i])
                                   + t_prev_layer[j][i] + t*q(x[i],y[j],t_i*t))
        for j in range(1,len(y)-1):
            y_x_layer[j][0] = (gamma_x(y[j], t_i*t)[0] * h_x- alpha_x()[0] * y_x_layer[j][1] ) / \
                              (beta_x()[0] * h_x- alpha_x()[0])
            y_x_layer[j][len(x)-1] = (gamma_x(y[j], t_i*t)[-1] * h_x - alpha_x()[-1] * y_x_layer[j][len(x)-2]) /\
                              (beta_x()[-1] * h_x - alpha_x()[-1])

        for i in range(len(x)):
            y_x_layer[0][i] = (gamma_y(x[i], t_i)[0] * h_y - alpha_y()[0] * y_x_layer[1][i]) /\
                      (beta_y()[0] * h_y - alpha_y()[0])
            y_x_layer[len(y)-1][i] = (gamma_y(x[i], t_i)[-1] * h_y - alpha_y()[-1] * y_x_layer[len(y)-2][i]) /\
                      (beta_y()[-1] * h_y - alpha_y()[-1])
        t_prev_layer = np.copy(y_x_layer)
        u.append(t_prev_layer)

    return u


xmin = -1
xmax = 1
ymin = -2
ymax = 2
h_x = 0.01
h_y = 0.02
Nt = 1
x = np.arange(xmin, xmax+ h_x, h_x)
y = np.arange(ymin, ymax+ h_y, h_y)
#print(q(x[36], h_x, y[71], h_y, 0.25, counter_for_cells))

t = Kurant_condition(h_x, h_y, a=1)
my = solution(x, y, h_x, h_y, Nt, r = 0.25)
mysol = np.array(my)
real_solution = resh(x,y,t)
print(1)

fig, ax = plt.subplots(2)
fig.set_size_inches((20,20))

# for i in range(2):
#     ax[i].imshow(my[i*49], extent=[xmin, xmax, ymin, ymax], interpolation='bilinear', origin='lower',
#                      cmap='jet')
#     ax[i].set_title(f't = {50*i}')
aa= mysol[1]
er= [abs(mysol[1][j][i] - resh(x[i],y[j],t)) for i in range(len(x)) for j in range(len(y))]
mm = max(er)

hx = [0.8, 0.4,0.02, 0.01, 0.005]
hy = [0.8, 0.4,0.02, 0.01, 0.005]
errs = []
for i in range(len(hx)):
    h_x = hx[i]
    h_y = hy[i]
    my = solution(x, y, h_x, h_y, Nt, r=0.25)
    mysol = np.array(my)
    real_solution = resh(x, y, t)
    er = [abs(mysol[1][j][i] - resh(x[i], y[j], t)) for i in range(len(x)) for j in range(len(y))]
    errs.append(max(er))
plt.figure()
plt.plot(hx, errs, color='yellow',)
plt.title(' график ошибки')
plt.legend()
plt.grid(True)
plt.show()