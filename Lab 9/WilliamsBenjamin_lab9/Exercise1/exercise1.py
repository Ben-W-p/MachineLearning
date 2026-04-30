import numpy as np
from scipy.optimize import minimize

data = np.genfromtxt('SVM_dualOptimization.csv', delimiter=',', skip_header=1)

X = data[:, :-1].astype(float)
y = data[:, -1].astype(float)

labels = np.unique(y)
if set(labels) == {0, 1}:
    y = np.where(y == 0, -1, 1)

n = len(y)

G = X @ X.T
Q = np.outer(y, y) * G

def objective(alpha):
    return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

def gradient(alpha):
    return Q @ alpha - np.ones_like(alpha)

constraints = [
    {
        'type': 'eq',
        'fun': lambda a: np.dot(a, y),
        'jac': lambda a: y
    }
]

bounds = [(0, None)] * n
alpha0 = np.zeros(n)

result = minimize(
    objective,
    alpha0,
    jac=gradient,
    bounds=bounds,
    constraints=constraints,
    method='SLSQP'
)

alpha = result.x

sv_idx = np.where(alpha > 1e-6)[0]
support_vectors = X[sv_idx]

w = np.sum((alpha * y)[:, None] * X, axis=0)

b_values = []
for s in sv_idx:
    b_s = y[s] - np.sum(alpha * y * G[:, s])
    b_values.append(b_s)

b = np.mean(b_values)

print('Support vectors:')
print(support_vectors)
print('\nw =', w)
print('b =', b)