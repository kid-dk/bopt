import os
import torch
import pandas as pd
torch.manual_seed(0)

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

# Define the data
data = pd.DataFrame({
    "PEG (g)": [0.0, 0.5, 1.0, 0.0],
    "NaCl (mg)": [0.0, 150.0, 150.0, 75.0],
    "Glycerol (ml)": [0.0, 0.0, 1.5, 0.0],
    "ESD": [430.700000, 252.821507, 76.741028, 352.021355],
    "RF": [0.040480, 0.846667, 0.930000, 0.660667]
})

# Define training data
train_x = torch.tensor(data.iloc[:, 0:3].values, **tkwargs)
train_y = torch.tensor(data.iloc[:, -2:].values, **tkwargs)

# Define bounds
bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 150.0, 2.0]], **tkwargs)

# Define reference point
ref_point = torch.tensor([0.0, 0.0], **tkwargs)

# Initialize the model
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, normalize
from botorch.fit import fit_gpytorch_mll

def initialize_model(train_X, train_y):
    train_X = normalize(train_X, bounds=bounds)
    train_Yvar = torch.full_like(train_y, 4e-2)
    model = SingleTaskGP(train_X, train_y, train_Yvar, outcome_transform=Standardize(m=2))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return mll, model

# Optimize acquisition function
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.logei import qLogNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

BATCH_SIZE = 4
NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4
MC_SAMPLES = 128 if not SMOKE_TEST else 16

def optimize_qehvi_and_get_observation(model, train_x, train_obj, sampler):
    """Optimizes the qEHVI acquisition function and returns a new candidate."""
    with torch.no_grad():
        pred = model.posterior(normalize(train_x, bounds)).mean
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
    acq_func = qLogNoisyExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        X_baseline=normalize(train_x, bounds),
        sampler=sampler,
    )
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], **tkwargs),  # Use normalized bounds
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={"batch_limit": 5, "maxiter": 200},
        sequential=True,
    )
    new_x = unnormalize(candidates.detach(), bounds=bounds)
    # Clamp the results to ensure they stay within bounds
    new_x = torch.clamp(new_x, min=bounds[0], max=bounds[1])
    return new_x

# Run the optimization
mll, model = initialize_model(train_x, train_y)
sampler = SobolQMCNormalSampler(sample_shape=torch.Size([MC_SAMPLES]))
new_x = optimize_qehvi_and_get_observation(model, train_x, train_y, sampler)

# Convert new_x to DataFrame
columns = data.columns[:3]
new_x_df = pd.DataFrame(new_x.cpu().numpy(), columns=columns)
print(new_x_df)