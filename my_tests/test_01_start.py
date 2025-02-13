import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoemulate.compare import AutoEmulate
from autoemulate.experimental_design import LatinHypercube
from autoemulate.simulations.epidemic import simulate_epidemic


seed = 42
np.random.seed(seed)

beta = (0.1, 0.5)  # lower and upper bounds for the transmission rate
gamma = (0.01, 0.2)  # lower and upper bounds for the recovery rate
lhd = LatinHypercube([beta, gamma])
X = lhd.sample(200)
y = np.array([simulate_epidemic(x) for x in X])

print(f"shapes: input X: {X.shape}, output y: {y.shape}\n")
print(f"X: {np.round(X[:3], 2)}\n")
print(f"y: {np.round(y[:3], 2)}\n")

transmission_rate = X[:, 0]
recovery_rate = X[:, 1]

em = AutoEmulate()
em.setup(X, y)
best_model = em.compare()

em.summarise_cv()

em.summarise_cv(model="GaussianProcess")

gp = em.get_model("GaussianProcess")
em.evaluate(gp)

gp_final = em.refit(gp)

seed = 42
np.random.seed(seed)

beta = (0.1, 0.5)  # lower and upper bounds for the transmission rate
gamma = (0.01, 0.2)  # lower and upper bounds for the recovery rate
lhd = LatinHypercube([beta, gamma])
X_new = lhd.sample(1000)
y_new = gp.predict(X_new)

transmission_rate = X_new[:, 0]
recovery_rate = X_new[:, 1]


em = AutoEmulate()
em.setup(
    X,
    y,
    param_search=True,
    param_search_type="random",
    param_search_iters=10,
    models=["SupportVectorMachines", "RandomForest"],
    n_jobs=-2,
)  # n_jobs=-2 uses all cores but one
em.compare()

from autoemulate.metrics import history_matching

pred_mean, pred_std = gp.predict(X, return_std=True)
pred_var = np.square(pred_std)  # Convert std to variance
expectations = (pred_mean, pred_var)
hist_match = history_matching(expectations=expectations, obs=[0, 1e-6], threshold=1.0)

print(f'The Not Rulled Out Points are {hist_match["NROY"]}')

print(X[hist_match["NROY"]])
from autoemulate.metrics import max_likelihood

result = max_likelihood(parameters=X, model=best_model, obs=[0, 10])

print(f"Indices of plausible regions: {result['optimized_params']}")


from autoemulate.simulations.projectile import simulate_projectile_multioutput

lhd = LatinHypercube(
    [(-5.0, 1.0), (0.0, 1000.0)]
)  # (upper, lower) bounds for each parameter
X = lhd.sample(100)
y = np.array([simulate_projectile_multioutput(x) for x in X])
X.shape, y.shape
X


em = AutoEmulate()
em.setup(X, y)

print(em.models[5])  # print the 5th model

### Example
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

mmscaler = MinMaxScaler()
pca = PCA(0.99)
kfold = KFold(n_splits=3, shuffle=False)

em = AutoEmulate()
em.setup(X, y, scale=True, scaler=mmscaler, cross_validator=kfold)
best_model = em.compare()


from autoemulate.metrics import history_matching

pred_mean, pred_std = best_model.predict(X, return_std=True)
pred_var = np.square(pred_std)  # Convert std to variance
expectations = (pred_mean, pred_var)
hist_match = history_matching(
    expectations=expectations, obs=[(1500, 53), (200, 80)], threshold=1.0
)

print(f'The Not Rulled Out Points are {hist_match["NROY"]}')
print(X[hist_match["NROY"]])

from autoemulate.metrics import max_likelihood

result = max_likelihood(parameters=X, model=best_model, obs=[(1500, 53), (200, 80)])

print(f"Indices of plausible regions: {result['optimized_params']}")
