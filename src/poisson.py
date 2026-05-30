import numpy as np
from scipy.linalg import solve_banded

def generate_data(
	N=64,
	num_samples=128,
	k_min=1,
	k_max=6,
	amp_decay=1.0,
	f_noise_std=0.0,
	dtype=np.float64,
	seed=None,
):
	"""
	Generates data (f, u) for 1D Poisson: -u''(x) = f(x), u(0)=u(1)=0 on [0,1].

	Discretization:
	  - grid includes boundaries: x_i = i*h, i=0..N-1, h=1/(N-1)
	  - unknowns are interior values u[1:-1] (size N-2)
	  - solve: (2 u_i - u} - u})/h^2 = f_i  for i=1..N-2
		i.e. A u_int = f_int with A tridiagonal: diag=2/h^2, offdiag=-1/h^2

	Forcing generation:
	  f(x) = sum} (a_k sin(2πkx) + b_k cos(2πkx)) / k}
	  with a_k,b_k ~ N(0,1)

	Parameters
	----------
	N : int
		Number of grid points including boundaries.
	batch_size : int
		Number of samples per batch.
	k_min, k_max : int
		Min/max Fourier mode numbers used to generate f.
	amp_decay : float
		Scales amplitudes ~ 1/k}; larger => smoother f.
	f_noise_std : float
		Std of additive Gaussian noise added to f values (per gridpoint).
	dtype : numpy dtype
	device : str or torch.device or None
		If None: returns numpy arrays (B,N).
		If not None: returns torch tensors on this device.
	seed : int or None
		Seed for reproducibility (uses its own RNG).

	Returns
	-------
	A pair (f,u) of size (num_samples x N, num_samples x N).
	"""
	assert N >= 3, "Need at least 3 grid points to have interior points."
	assert k_max >= k_min >= 1

	rng = np.random.default_rng(seed)

	# Grid
	x = np.linspace(0.0, 1.0, N, dtype=dtype)
	h = dtype(1.0) / dtype(N - 1)

	# Precompute trig basis for speed: shapes (K, N)
	ks = np.arange(k_min, k_max + 1)
	K = len(ks)
	# (K, N)
	sin_basis = np.sin((2.0 * np.pi * ks[:, None]) * x[None, :]).astype(dtype)
	cos_basis = np.cos((2.0 * np.pi * ks[:, None]) * x[None, :]).astype(dtype)
	scale = (ks.astype(dtype) ** (-dtype(amp_decay))).astype(dtype)  # (K,)

	# Build banded tridiagonal for solve_banded: shape (3, n) where n=N-2 interior
	n_int = N - 2
	main = (2.0 / (h * h)) * np.ones(n_int, dtype=dtype)
	off  = (-1.0 / (h * h)) * np.ones(n_int - 1, dtype=dtype)
	# ab[0] upper diagonal (length n-1), ab[1] main (length n), ab[2] lower (length n-1)
	ab = np.zeros((3, n_int), dtype=dtype)
	ab[0, 1:] = off
	ab[1, :]  = main
	ab[2, :-1]= off

	# Sample Fourier coefficients: (B,K)
	a = rng.standard_normal((num_samples, K), dtype=dtype)
	b = rng.standard_normal((num_samples, K), dtype=dtype)
	a *= scale[None, :]
	b *= scale[None, :]

	# f = a @ sin_basis + b @ cos_basis => (B,N)
	f = a @ sin_basis + b @ cos_basis

	if f_noise_std > 0:
		f = f + rng.normal(0.0, f_noise_std, size=f.shape).astype(dtype)

	# Solve for u interior
	f_int = f[:, 1:-1]  # (B, N-2)
	# solve_banded can solve multiple RHS if RHS is (n, nrhs), so transpose
	u_int_T = solve_banded((1, 1), ab, f_int.T)  # (n_int, B)
	u_int = u_int_T.T.astype(dtype)              # (B, n_int)

	# Pad boundaries with zeros
	u = np.zeros((num_samples, N), dtype=dtype)
	u[:, 1:-1] = u_int

	# * 100 as a very simple normalization to get solutions closer to stddev==1
	return f, 100.0 * u, x
