import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.animation as animation
import math

num_of_records = 200
x0 = np.repeat(1, num_of_records)
x1 = np.random.uniform(low=0.0, high=1.0, size=(num_of_records,))
y = np.matrix([4 + 3 * x1 + np.random.uniform(low=0.0, high=1.0, size=(num_of_records,))]).transpose()
mat_x = np.matrix([x0, x1]).transpose()
learning_rate = 0.01
g_theta_0, g_theta_1 = 1, 1
g_cur_cost = math.inf

def cal_cost(theta0, theta1):
    mat_theta = np.matrix([[theta0], [theta1]])
    y_predict = mat_x.dot(mat_theta)
    y_diff = y_predict - y
    cost = np.sum( np.power( y_diff, 2 ) ) / (2 * num_of_records)
    return cost

def cal_y(theta0, theta1):
    mat_theta = np.matrix([[theta0], [theta1]])
    y_predict = mat_x.dot(mat_theta)
    return np.transpose(y_predict).tolist()[0]

def run_gradient_descent(theta0, theta1):
    mat_theta = np.matrix([[theta0], [theta1]])
    y_predict = mat_x.dot(mat_theta)
    y_diff = y_predict - y

    # Theta 0 Calculation
    theta_0_offset = np.sum(y_diff) * learning_rate / num_of_records
    new_theta_0 = theta0 - theta_0_offset

    # Theta 1 Calculation
    y_theta_1_diff = y_diff
    for idx in range(0, num_of_records):
        y_theta_1_diff[idx] = y_theta_1_diff[idx] * x1[idx]
    theta_1_offset = np.sum(y_theta_1_diff) * learning_rate / num_of_records
    new_theta_1 = theta1 - theta_1_offset

    return(new_theta_0, new_theta_1)

def animate(i):
    global g_theta_0, g_theta_1,g_cur_cost

    if g_cur_cost > 0.05:
        g_theta_0, g_theta_1 = run_gradient_descent(g_theta_0, g_theta_1)
        g_cur_cost = cal_cost(g_theta_0, g_theta_1)

    line.set_ydata(cal_y(g_theta_0, g_theta_1))  # update the data.
    return line,

def init():  # only required for blitting to give a clean slate.
    line.set_ydata(cal_y(g_theta_0, g_theta_1))
    return line,

fig, ax = plt.subplots()
line, = ax.plot(x1, cal_y(g_theta_0, g_theta_1), color = "red")

if __name__ == "__main__":
    plt.scatter(x1.tolist(), y.tolist())
    plt.title('Data Chart')
    plt.xlabel('x')
    plt.ylabel('y')
    anim = animation.FuncAnimation(fig, animate, interval=10)
    plt.show()

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	