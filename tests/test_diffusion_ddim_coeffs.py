"""Unit tests for ``DiffusionInferenceEngine._ddim_coeffs``.

This pins a clean-checkout regression where ``_ddim_coeffs`` used
``torch.where`` / ``torch.as_tensor`` but only imported ``torch`` under
``TYPE_CHECKING``. In that state, the first denoising step raised
``NameError: name 'torch' is not defined`` at runtime.

The test keeps the setup minimal by mocking only the scheduler fields that
``_ddim_coeffs`` actually reads.
"""

from types import SimpleNamespace

import torch

from tests.diffusion_test_utils import assert_tensors_close, load_diffusion_module

DiffusionInferenceEngine = load_diffusion_module(
    "diffusion_engine"
).DiffusionInferenceEngine


def test_ddim_coeffs_runtime_torch_import_regression():
    """``_ddim_coeffs`` must run at runtime without relying on TYPE_CHECKING."""
    engine = DiffusionInferenceEngine(model_path="unused")
    engine.scheduler = SimpleNamespace(
        config=SimpleNamespace(num_train_timesteps=1000),
        num_inference_steps=10,
        alphas_cumprod=torch.linspace(0.1, 0.9, 1000),
        final_alpha_cumprod=torch.tensor(0.05),
    )

    timestep = torch.tensor(999)
    alpha_prod_t, alpha_prod_t_prev, beta_prod_t = engine._ddim_coeffs(timestep)

    assert torch.is_tensor(alpha_prod_t)
    assert torch.is_tensor(alpha_prod_t_prev)
    assert torch.is_tensor(beta_prod_t)
    assert_tensors_close(alpha_prod_t, engine.scheduler.alphas_cumprod[999])
    assert_tensors_close(alpha_prod_t_prev, engine.scheduler.alphas_cumprod[899])
    assert_tensors_close(beta_prod_t, 1 - engine.scheduler.alphas_cumprod[999])
