"""Guard: the detection-delay benchmark must reject action-masked envs.

``detection_delay_env``'s partner-swap intervention plays a *fixed* scripted action for the
whole post-swap window and scores on an all-ones mask -- only well-defined on an unmasked /
fixed-action-set env. On a masked env (SMAX: legal actions change as units die / come into
range) those numbers are not a valid benchmark, so the function fails fast rather than
silently producing junk. This pins that contract (the unmasked envs are covered by the
Overcooked / MPE smoke runs in ``run_benchmark.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# ``scripts/`` is not a package (modules are run as ``python scripts/foo.py``); put it on the
# path so ``detection_benchmark`` and its sibling ``drift_demo`` import, exactly as the
# scripts do at runtime.
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.mark.slow
def test_detection_delay_env_rejects_masked_env():
    """SMAX (a real action-masked env) is rejected *before* any training, with a message
    pointing at the unmasked-env requirement. Marked ``slow`` because it builds SMAX."""
    import detection_benchmark as db
    from conformal_marl.envs.registry import make_env

    # Precondition: SMAX genuinely masks actions (else this test proves nothing).
    _, spec = make_env("SMAX", map_name="3m")
    assert spec.masks_actions

    # A calib dict is required by the signature but the guard fires before it is read.
    dummy_calib = {"aes_ref_mean": 0.0, "aes_ref_std": 1.0, "aes_threshold": 1.0}
    with pytest.raises(NotImplementedError, match="action-masked"):
        db.detection_delay_env(
            dummy_calib, delta=0.05, seeds=[0], pre_len=4, post_len=4,
            env_name="SMAX", hidden_dim=32, map_name="3m",
        )
