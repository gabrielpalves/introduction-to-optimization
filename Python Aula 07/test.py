import numpy as np

x1 = np.arange(-20,20,1)
print(x1.shape)

x = np.array([1, -1])
h = x[0]+x[1]-4
dh = np.array([1, 1])
g = np.array([])
f = np.array([[1,2,3,4],[3,4,5,6]])

print(x.shape, h.shape, dh.shape, f.shape)
print(x.size, h.size, dh.size, f.size)

if h.size == 0:
    print('not even h')

if g.size == 0:
    print('not even g')

if not h.shape:
    print('not')
else:
    print('ok')

if not x.shape:
    print('not x')
    
if len(h.shape) == 0:
    print('length is zero')


if isinstance(x, list):
    x = np.array(x)

h = x[0]+x[1]-4
dh = np.array([1, 1])

# divide o resultado de 'h' pelo negativo da derivada