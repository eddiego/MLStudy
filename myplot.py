import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-1. , 6. , 0.2)
y1 = 4*x - 10
y2 = np.exp(x-2)

#plt.plot(x, x*x - 4*x + 5 ,'b-', x, 2*x-4, 'r-', [3],[2],'ro')
plt.plot([1,2,3,4],[2,4,6,8],'ro', x,y1,'b--', x,y2,'g-')

plt.axis([-2,8,0,10]);
plt.title('my 2d plot')
plt.xlabel('x label')
plt.ylabel('y label')
plt.text(1,8, 'my text', fontsize=12)
plt.annotate('this point', xy=(3, 6), xytext=(1, 5),
            arrowprops=dict(facecolor='black', shrink=0.01))

plt.show()

