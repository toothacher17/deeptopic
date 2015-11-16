import math

# approximate implementation of digamma function
def digamma(x):
    if int(x) == 0:
        x = 0.0001
    xp2 = x + 2
    return math.log(xp2) - (6*x+13)/(12*xp2*xp2) - (2*x+1)/(x*x+x)
