from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import numpy as np

 
# define the true objective function
def objective(x, a, b, c, d, e, f, g, h, i):
 return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + (f * x**6) + (g * x**7) + (h * x**8) + i
 
# load the dataset
#url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/longley.csv'
url = '3-3.csv'
dataframe = read_csv(url, header=None)

dataRAW = dataframe.values
data = dataRAW[dataRAW[:,0] >= 10]

# choose the input and output variables
#x, y = data[:, 4], data[:, -1]
x, y = data[:, 1], data[:, 0]
#x, y = data['torque'], data['angle']

# curve fit
popt, _ = curve_fit(objective, x, y)

# summarize the parameter values
a, b, c, d, e, f, g, h, i = popt
####print('y = %.5f * x + %.5f' % (a, b))

# plot input vs output
plt.scatter(x, y, linewidths=0.5, marker= "." )

# define a sequence of inputs between the smallest and largest known inputs
x_line = arange(min(x), max(x), 1)

# calculate the output for the range
y_line = objective(x_line, a, b, c, d, e, f, g, h, i)

# create a line plot for the mapping function
plt.plot(x_line, y_line, '--', color='red')

# make the plot interactive and get the coordinates of two points clicked by the user
Click_points = plt.ginput(2)
print('You clicked:', Click_points)

# find the x values for the y coordinates of the clicked points on the curve
#for slope_point in Click_points:
#    y_value_Cl = Click_points[1]
#    x_value_Cl = fsolve(lambda x : objective(x, a, b, c, d, e, f, g, h, i) - y_value_Cl, 0)
#    print(f'The x value on the curve for y = {y_value_Cl} is {x_value_Cl}')

plt.xlim(0, max(x)*1.05)
plt.ylim(0, max(y)*1.05)
plt.show()

# find the slope between the clicked points
#x1, y1 = Click_points[0]
#x2, y2 = Click_points[1]
#slope = (y2 - y1) / (x2 - x1)
#print(f'The slope between the clicked points is {slope}')

# find the slope between the clicked points
cpx1, cpy1 = Click_points[0]
cpx2, cpy2 = Click_points[1]

PolyN = np.poly1d([h, g, f, e, d, c, b, a, i])

cp_ey1 = PolyN(cpx1)
cp_ey2 = PolyN(cpx2)

slope = (cp_ey2 - cp_ey1) / (cpx2 - cpx1)
print(f'The slope between the clicked points is {slope}')

# create a new figure for the line plot
plt.figure()

# calculate the y-intercept of the line
#intercept = y1 - slope * x1
Slope_intercept = cp_ey1 - slope * cpx1

# calculate the y values of the line
y_line_slope = slope * x_line + Slope_intercept

# plot the line
#plt.plot(x_line, y_line_slope, '--', color='green')
#plt.scatter(x, y, linewidths=0.5, marker= "." )
#plt.plot(x_line, y_line, '--', color='red')
#plt.text(0.95, 0.05, f'Slope: {slope}', transform=(1,1), verticalalignment='bottom', horizontalalignment='right')
#plt.xlim(0, max(x)*1.025)
#plt.ylim(0, max(y)*1.1)
#
#plt.show()

print(f"a={a}")
print(f"b={b}")
print(f"c={c}")
print(f"d={d}")
print(f"e={e}")
print(f"f={f}")
print(f"g={g}")
print(f"h={h}")
print(f"i={i}")
print(f"Slope={slope}")

# Define the polynomial and the slope
polynomial = a*x + b*x**2 + c*x**3 + d*x**4 + e*x**5 + f*x**6 + g*x**7 + h*x**8 + i

YT_slope = slope * 2 /3

print("Slope:", slope)
print("YT_slope:", YT_slope)

#PolyN = np.poly1d([h, g, f, e, d, c, b, a, i]) 
print("Polynomial function, f(x):\n", PolyN) 
  
# calculating the derivative 
derivative = PolyN.deriv() 
print("Derivative, f(x)'=", derivative) 

# Define the function for the derivative minus YT_slope
def func(x):
    return derivative(x) - YT_slope

# Initial guess for x
x0 = 400

# Use fsolve to find the x value
YT_Deg = fsolve(func, x0)
YT_Torque = PolyN(YT_Deg)

print(f"The Yield Torque (T2) = {YT_Torque}Nm @ {YT_Deg} Degrees")
  
# calculates the derivative of after  
# given value of x 
#print("When x={YT_slope}  f(x)'=", derivative(YT_slope)) 

Yt_ey1 = PolyN(cpx1)
YT_intercept = YT_Torque - YT_slope * YT_Deg
YT_line_slope = YT_slope * x_line + YT_intercept

# plot the line
plt.plot(x_line, y_line_slope, '-', color='violet')
plt.plot(x_line, YT_line_slope, '-', color='green')
plt.plot(YT_Torque,YT_Deg, marker='o', markersize=100, label='line & marker')
plt.scatter(x, y, linewidths=0.5, marker= "." )
plt.plot(x_line, y_line, '--', color='red')
plt.text(0.99 * max(x), 0.5, f'T-Θ: {slope}° \n 2/3 T-Θ: {YT_slope}° \n Yield Torque (T2):{YT_Torque}Nm @ {YT_Deg}°', verticalalignment='bottom', horizontalalignment='right')
#plt.text(0.75* max(x), 0.5, f'Slope: {slope}')
plt.xlim(0, max(x)*1.025)
plt.ylim(0, max(y)*1.1)

plt.show()
