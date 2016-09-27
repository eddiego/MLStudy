from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#x = np.random.random(10)

xi = np.arange(0,10)
#A = np.array([ xi, np.ones(9)])
y = [19, 20, 20.5, 21.5, 22, 23, 23, 25.5, 24, 25]

"""
xss = sum([i ** 2 for i in xi])
yss = sum([i ** 2 for i in y])
theta0 = np.arange(5, 35, 0.25)
theta1 = np.arange(-5, 5, 0.25)
T0, T1 = np.meshgrid(theta0,theta1)
J = (10*T0**2 + xss*T1**2 + yss + 2*sum(xi)*T0*T1 - 2*sum(xi*y)*T1 -2*sum(y)*T0)/20

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(T0,T1,J,rstride=2,cstride=2,cmap=cm.RdPu,linewidth=1,antialiased=True)
ax.set_xlabel('theta0')
ax.set_ylabel('theta1')
ax.set_zlabel('J')
plt.show()
"""
print np.polyfit(xi,y,1,True)

'''
slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)

print 'r value', r_value
print  'p_value', p_value
print 'standard deviation', std_err
print slope 
print intercept
line = slope*xi+intercept
'''
"""
f,(ax1,ax2) = plt.subplots(1,2)
ax1.plot(xi,y,'o')
ax1.axis([-1,11,15,26])
ax2.plot(xi,line,'r-',xi,y,'o')
ax2.axis([-1,11,15,26])
plt.show()
"""
