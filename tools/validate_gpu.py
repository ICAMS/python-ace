"""Run PACE fitting tests and verify native energy/force/gradient placement."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import tensorpotential
import tensorflow as tf
import pytest
from tensorpotential.compat.pace.tensorpot import TensorPotential

physical = tf.config.list_physical_devices("GPU")
assert len(physical) == 1, f"Expected exactly one selected GPU, found {physical}"
for device in physical:
    tf.config.experimental.set_memory_growth(device, True)
tf.config.set_soft_device_placement(False)
with tf.device("/GPU:0"):
    product = tf.linalg.matmul(tf.ones((128, 128), tf.float64), tf.ones((128, 128), tf.float64))
assert "GPU:0" in product.device, product.device
assert float(product[0, 0]) == 128.0
print("GPU_PREFLIGHT", json.dumps({
    "tensorflow": tf.__version__, "tensorpotential": tensorpotential.__version__,
    "details": tf.config.experimental.get_device_details(physical[0]),
    "matmul_device": product.device,
}), flush=True)

originals = {name: getattr(TensorPotential, name) for name in ("native_fit", "external_fit")}
placements = []

def check_result(result):
    devices = {name: value.device for name, value in zip(("loss", "gradient", "energy", "forces"), result)}
    for name in ("gradient", "energy", "forces"):
        assert "GPU:0" in devices[name], f"PACE {name} fell back to {devices[name]}"
    if not placements:
        print("PACE_NATIVE_PLACEMENT", json.dumps(devices), flush=True)
    placements.append(devices)
    return result

def checked_fit(name):
    def call(self, *args, **kwargs):
        return check_result(originals[name](self, *args, **kwargs))
    return call

for name in originals:
    setattr(TensorPotential, name, checked_fit(name))
args = sys.argv[1:] or ["tests/", "--runtensorpot", "-m", "tensorpot", "-q"]
status = pytest.main(args)
summary = {
    "pytest_exit_code": int(status), "fitting_calls_on_gpu": len(placements),
    "first_placement": placements[0] if placements else None,
    "peak_gpu_memory": tf.config.experimental.get_memory_info("GPU:0"),
}
print("GPU_VALIDATION_RESULT", json.dumps(summary), flush=True)
assert placements, "No PACE fitting calls were exercised"
sys.exit(status)
