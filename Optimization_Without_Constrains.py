import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections as dic
from collections import OrderedDict
import math



# Utillity function
def f(x,y):
    return (x*y-x**2*y-x*y**2)/8

# Derivatives of utillity function
def dfx(x,y):
    return (y-2*x*y-y**2)/8
def dfy(x,y):
    return (x-2*x*y-x**2)/8

# Function for calculating gradient norm
def gradientNorm(x,y):
    return ((dfx(x,y)**2)+(dfy(x,y)**2))**(1/2.0)

# Gradient descent algorithm
def gradientDescent(x,y,length):
    i = 0
    pointsX = []
    pointsY = []
    while gradientNorm(x,y) > 0.0001:
        xadd = length*dfx(x,y)
        yadd = length*dfy(x,y)
        x += xadd
        y += yadd
        i += 1
        pointsX.append(x)
        pointsY.append(y)
        # print (x,y,i,f(x,y))
    return x,y,i,f(x,y), pointsX, pointsY

# Steepest descent algorithm
def fastestDescent(x,y):
    i = 0
    pointsX = [x]
    pointsY = [y]
    while gradientNorm(x,y) > 0.0001:
        length = argmax(x,y)
        xadd = length*dfx(x,y)
        yadd = length*dfy(x,y)
        x += xadd
        y += yadd
        i += 1
        pointsX.append(x)
        pointsY.append(y)
        print (x,y,i,f(x,y), length)
        # print(length)
    return x,y,i,f(x,y),pointsX,pointsY

# Modified utility function for steepest descent
def S(x,y,q):
    return ((x+q*dfx(x,y))*(y+q*dfy(x,y))-(x+q*dfx(x,y))**2*(y+q*dfy(x,y))-(x+q*dfx(x,y))*(y+q*dfy(x,y))**2)/8

# Function for maximazing length of the step for steepest descent
def argmax(x,y):
    i = 0
    r = 9
    l = 0
    L = r-l
    e = 0.0001
    qm = 0
    while L > e:
        L = r-l
        qm = (l+r)/2
        q1 = l+L/4
        q2 = r-L/4
        if S(x,y,q1) > S(x,y,qm):
            r = qm
            qm = q1
        elif S(x,y,q2) > S(x,y,qm):
            l = qm
            qm = q2
        else:
            l = q1
            r = q2
        i += 1
        print(i)
    return qm

# Function for creating a 3 pointed simplex
def simplex(x,y,a):

    sigma1 = ((math.sqrt(3) + 1)/2*math.sqrt(2))*a
    sigma2 = ((math.sqrt(3) - 1)/2*math.sqrt(2))*a
    x0 = [x,y]
    x1 = [x+sigma2, y+sigma1]
    x2 = [x+sigma1, y+sigma2]
    points = [x0,x1,x2]
    return points

# result = gradientDescent(0,0,1)
# pointsX = result[4]
# pointsY = result[5]
# print(result[0],result[1],result[2],result[3])

# result = fastestDescent(0.7,0.8)
# print(result[0],result[1],result[2])
# pointsX = result[4]
# pointsY = result[5]

# Finding min value of simplex
def findMinPoint(points):
    ptemp = {}
    p={}
    i = 0
    for pert in points:
        key = f(points[i][0],points[i][1])
        if(ptemp.get(key)):
            key = key + 0.000001
        ptemp[key] = points[i]
        i += 1
    p = dict(sorted(ptemp.items()))
    return p[list(p.keys())[0]], p[list(p.keys())[1]], p[list(p.keys())[2]]

# filtering out the min value point from simplex
def filterPoints(points, minpertice):
    result = []
    for p in points:
        if p != minpertice:
            result.append(p)
    return result

# calculating the center of the simplex
def center(p):
    return [(p[0][0]+p[1][0])/2,(p[0][1]+p[1][1])/2]

# calculating new point of the simplex
def newPoint(c, Xh, Xg, Xl,a):
    o = 1
    newp = [Xh[0]+((1+o)*(c[0]-Xh[0])),Xh[1]+((1+o)*(c[1]-Xh[1]))]
    if (f(newp[0],newp[1]) > f(Xg[0],Xg[1]) and f(newp[0],newp[1]) < f(Xl[0],Xl[1])):
        o = 1
        a = a*1
        Z = [Xh[0]+((1+o)*(c[0]-Xh[0])),Xh[1]+((1+o)*(c[1]-Xh[1]))]
    elif (f(newp[0],newp[1]) > f(Xl[0],Xl[1])):
        o = 2
        a = a*2
        if(f(-(Xh[0])+(1+o)*c[0],-(Xh[1])+(1+o)*c[1]) > f(newp[0],newp[1])):
            Z = [Xh[0]+((1+o)*(c[0]-Xh[0])),Xh[1]+((1+o)*(c[1]-Xh[1]))]
        else:
            Z = newp
    elif (f(newp[0],newp[1]) < f(Xh[0],Xh[1])):
        o = -0.5
        a = a*0.5
        Z = [Xh[0]+((1+o)*(c[0]-Xh[0])),Xh[1]+((1+o)*(c[1]-Xh[1]))]
    elif (f(newp[0],newp[1]) > f(Xh[0],Xh[1]) and f(newp[0],newp[1]) < f(Xg[0],Xg[1])):
        o = 0.5
        a = a*0.5
        Z = [Xh[0]+((1+o)*(c[0]-Xh[0])),Xh[1]+((1+o)*(c[1]-Xh[1]))]
    else:
        Z=newp
    return Z,a

# Deformed simplex algorithm
def simplexAlgorithm(points,a):
    result = findMinPoint(points)
    minValuePoint = result[0]
    midValuePoint = result[1]
    maxValuePoint = result[2]
    p = filterPoints(points,minValuePoint)
    c = center(p)
    result = newPoint(c,minValuePoint,midValuePoint,maxValuePoint,a)
    p.append(result[0])
    a = result[1]
    return p,a, maxValuePoint


points = simplex(0,0,0.1)

a = 1
i = 0
pointsX = []
pointsY = []
while (a > 0.0001):
    result = simplexAlgorithm(points,a)
    points = result[0]
    a = result[1]
    mv = result[2]
    pointsX.append(mv[0])
    pointsY.append(mv[1])
    i = i+1
    print(mv, "  ", a, "  ", f(mv[0],mv[1]), "   ", i)



# plotting the points 

# x = np.arange(-1, 1, 0.25)
# y = np.arange(-1, 1,0.25)
# X, Y = np.meshgrid(x,y)

# F = (X*Y-X**2*Y-X*Y**2)/8

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, F,color="white", alpha=0.7)

# ax.scatter(pointsX,pointsY,c="#DC143C",depthshade=False)

# plt.show()
