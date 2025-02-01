from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import os

app = Flask(__name__)
CORS(app)  # Enable CORS

torch.manual_seed(0)
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
SMOKE_TEST = os.environ.get("SMOKE_TEST")

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    initial_data = data['initial_data']
    batch_size = data['batch_size']

    # Convert initial data to DataFrame
    df = pd.DataFrame(initial_data, columns=["PEG (g)", "NaCl (mg)", "Glycerol (ml)", "ESD", "RF"])
    train_x = torch.tensor(df.iloc[:, 0:3].values, **tkwargs)
    train_y = torch.tensor(df.iloc[:, -2:].values, **tkwargs)

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
            q=batch_size,
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
    columns = df.columns[:3]
    new_x_df = pd.DataFrame(new_x.cpu().numpy(), columns=columns)

    # Return the result as JSON
    return jsonify({
        'status': 'success',
        'result': new_x_df.to_dict(orient='records')
    })

if __name__ == '__main__':
    app.run(debug=True)