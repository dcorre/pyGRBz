import numpy as np
import matplotlib.pyplot as plt
import sys

folder = str(sys.argv[1])
#folder = 'bessel'
#folder = 'stroemgren'
#folder = 'sloan'
if folder == 'bessel':
    bands = ['U','B','V','R','I']
elif folder == 'sloan':
    bands = ['u','g','r','i','z']
elif folder == 'stroemgren':
    bands = ['u','v','b','y']

for band in bands:
    print (band)
    xx = []
    yy = []
    line = 0
    filename=folder+'_'+band+'.txt'
    with open('%s/%s' % (folder,filename), 'r') as file:
         for row in file:
              if line > 1:
                   a, b = row.split()
                   xx.append(float(a))
                   yy.append(float(b))
              line+=1 

    ang=np.array(xx)    # Angstrom
    Trans=np.array(yy)         # %

    plt.plot(ang,Trans,label=band)
plt.ylim(0.,100.)
plt.xlabel(r'wavelenghts (nm)')
plt.ylabel('Transmission ')
plt.legend()
plt.title('%s' % folder)
plt.grid(True)
plt.savefig('filter_%s.png' % folder)
plt.show()


