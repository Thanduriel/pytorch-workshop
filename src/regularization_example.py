import numpy as np
import matplotlib.pyplot as plt
import torch
import neuralnets

class Polynomial(torch.nn.Module):
	def __init__(self, degree):
		super(Polynomial, self).__init__()
		self.coefficients = torch.nn.Parameter(torch.rand(degree))
		self.degree = degree

	def forward(self, x):
		result = self.coefficients[0]
		x_pow = x
		for i in range(1, self.degree):
			result = result + x_pow * self.coefficients[i]
			x_pow = x_pow * x

		return result
	
def print_coords(xs,ys):
	for x, y in zip(xs.tolist(), ys.tolist()):
		print(f'({x}, {y})')


n = 13
np.random.seed(42)
points = np.random.random((2,n)).astype(np.float32) * 1.0
points[0] = np.linspace(0.0, 1.0, n) + np.random.random(n) * 0.05
points[0,0] = 0.0
points[0,-1] = 1.0
print_coords(points[0], points[1])

poly = np.polynomial.Polynomial.fit(points[0], points[1], n-1)

poly_low = np.polynomial.Polynomial.fit(points[0], points[1], 5)
poly_reg = np.polynomial.Polynomial.fit(points[0], points[1], n)

#poly_nn = Polynomial(n)
torch.manual_seed(1353)
poly_nn = neuralnets.make_simple_nn(1, 4, 1, 1)
points_torch = torch.tensor(points)

#optimizer = torch.optim.AdamW(poly_nn.parameters(), lr=0.1, weight_decay=0.0)#
optimizer = torch.optim.LBFGS(poly_nn.parameters(), lr=0.01) # , lr=0.1, weight_decay=0.0

def mse_reg(a, b, params, l):
	loss = torch.nn.functional.mse_loss(a, b)
	for param in params:
		loss += l * torch.sum(torch.square(param))

	return loss

loss_fn = torch.nn.MSELoss()

def closure():
	optimizer.zero_grad()
	outputs = poly_nn(points_torch[0].unsqueeze(1)).squeeze(1)
	#loss = mse_reg(outputs, points_torch[1], poly_nn.parameters(), 0.0001)
	loss = loss_fn(outputs, points_torch[1])
	loss.backward()
	return loss

for epoch in range(128):
	print(optimizer.step(closure))

x = np.linspace(0.0, 1.0, 64)
plt.plot(points[0], points[1], linestyle='', marker='*', label='samples')

#plt.plot(x, poly(x), label='fit')
#plt.plot(x, poly_low(x), label='low order')
#plt.plot(x, poly_reg(x), label='regularized')

#for i in range(n, n+4):
#	poly_reg = np.polynomial.Polynomial.fit(points[0], points[1], i)
#	plt.plot(x, poly_reg(x), label=f'{i}')

#data = np.stack([x, poly(x), poly_low(x), poly_reg(x)], axis=1)
#np.savetxt('poly_data.txt', data)

with torch.inference_mode():
	y = poly_nn(torch.tensor(x.astype(np.float32)).unsqueeze(1)).squeeze(1)
	plt.plot(x, y, label='NN')
	data = np.stack([x, y.numpy()], axis=1)
	np.savetxt('data_small.txt', data)

plt.legend()
plt.show()
