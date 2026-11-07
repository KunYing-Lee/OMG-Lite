"""Initial-noise policy for the frozen OMG-Lite generator.

The actor maps the canonical 10x38 raw-reference history to a diagonal
Gaussian over the 100x38 initial diffusion noise.  The generator and tracker
remain frozen; this package contains only the deployment-side actor, the
sampling helper and the 20-step PyTorch planner used by the noise service.
"""

from omg.noise.actor import NoiseActor
from omg.noise.sampling import sample_action_noise

__all__ = ["NoiseActor", "sample_action_noise"]
