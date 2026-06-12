import numpy as np

from se3force.data.rmd17 import load_rmd17_npz, rmd17_samples_from_npz


def test_rmd17_loader_reads_fake_npz_with_alias_keys(tmp_path):
    path = tmp_path / "fake_rmd17.npz"
    coords = np.random.randn(4, 3, 3).astype("float32")
    forces = np.random.randn(4, 3, 3).astype("float32")
    energies = np.random.randn(4).astype("float32")
    atomic_numbers = np.array([1, 6, 8], dtype="int64")
    np.savez(path, coords=coords, forces=forces, energies=energies, atomic_numbers=atomic_numbers)
    arrays = load_rmd17_npz(path)
    assert arrays["positions"].shape == (4, 3, 3)
    assert arrays["forces"].shape == (4, 3, 3)
    assert arrays["atomic_numbers"].tolist() == [1, 6, 8]
    samples = rmd17_samples_from_npz(path, "fake")
    assert samples[0]["z"].tolist() == [1, 6, 8]
    assert samples[0]["forces"].shape == (3, 3)
