import matplotlib.pyplot as plt

f = open("perp")

x = []
y = []
it = 0
for l in f:
    if it % 10 == 0:
        x.append(it)
        y.append(float(l.strip().split(" ")[-1]))
    it += 1
f.close()
plt.plot(x, y)
plt.show()
