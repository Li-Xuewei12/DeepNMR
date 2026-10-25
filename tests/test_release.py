"""Small regression tests; no downloaded weights or datasets are needed."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import pandas as pd
import torch
from deepnmr.evaluation import assigned_metrics, cluster_positions, peak_errors, gaussian_score
from stage_delta.data import collate_graphs
from stage_delta.model import NucleusShiftModel
from deepnmr.overlay import run_epoch
from torch.utils.data import DataLoader
from stage_j.data import collate_couplings
from predict import attach_references


def graph(uid="m"):
    return dict(molecule_uid=uid, conformer_uid=uid, sample_weight=1., heavy_atom_count=1,
                atomic_number=torch.tensor([6, 1]), formal_charge=torch.tensor([5, 5]),
                chirality=torch.zeros(2, dtype=torch.long), aromatic=torch.zeros(2, dtype=torch.long),
                bond_index=torch.tensor([[0, 1]]), bond_type=torch.tensor([0]),
                coordinates=torch.tensor([[0., 0., 0.], [1., 0., 0.]]),
                target_nucleus=torch.tensor([1, 0]), target_shift=torch.tensor([20., 2.]))


class ToyOverlay(torch.nn.Module):
    """One trainable value makes the accumulated gradient independently checkable."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.shift = type("Nucleus", (), {"nucleus_code": 0})()

    def forward(self, batch):
        return self.weight.expand(int(batch["target_nucleus"].eq(0).sum()))


class ReleaseTests(unittest.TestCase):
    def test_metrics_and_empty_sets(self):
        result = assigned_metrics([1, 3], [2, 1])
        self.assertEqual(result["MAE"], 1.5)
        self.assertAlmostEqual(result["RMSE"], np.sqrt(2.5))
        self.assertIsNone(assigned_metrics([1, 1], [1, 2])["R2"])
        self.assertEqual(peak_errors([1, 2], [], 10)["set_error"], 10)
        self.assertEqual(peak_errors([], [], 1)["set_error"], 0)
        self.assertAlmostEqual(gaussian_score([1, 2], [2, 1], .1), 1)
        np.testing.assert_allclose(cluster_positions([1., 1.02, 3.], .03), [1.01, 3.])
        for function, args in [(assigned_metrics, ([1], [float("nan")])),
                               (peak_errors, ([1], [1], -1)),
                               (gaussian_score, ([1], [1], 0)),
                               (cluster_positions, ([1], -1))]:
            with self.assertRaises(ValueError):
                function(*args)

    def test_unlabeled_inference_and_mixed_nuclei(self):
        item = graph()
        item.pop("target_shift")
        batch = collate_graphs([item], require_targets=False)
        self.assertEqual(batch["target_atom_index"].tolist(), [0, 1])
        model = NucleusShiftModel("1H", hidden_dim=8, topology_layers=1,
                                  spatial_layers=1, rbf_count=4, dropout=0.).eval()
        output = model(batch)
        self.assertEqual(output["target_mask"].tolist(), [False, True])
        self.assertEqual(tuple(output["prediction_ppm"].shape), (1,))
        self.assertTrue(torch.isfinite(output["prediction_ppm"]).all())

    def test_overlay_mask_and_partial_accumulation(self):
        loader = DataLoader([graph(str(i)) for i in range(3)], batch_size=1,
                            collate_fn=collate_graphs)
        model = ToyOverlay()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        metrics, rows = run_epoch(model, loader, "cpu", optimizer, amp=False, accumulation_steps=2)
        # Two L1 updates of +0.1, including the single-batch final window.
        self.assertAlmostEqual(float(model.weight), .2, places=5)
        self.assertEqual(metrics["optimizer_steps"], 2)
        self.assertEqual(rows.atom_index_conformer.tolist(), [1, 1, 1])
        self.assertEqual(rows.target_ppm.tolist(), [2., 2., 2.])

    def test_placeholder_targets_cannot_enter_training(self):
        item = graph()
        item["target_kind"] = "zero_inference_placeholder"
        self.assertTrue(torch.isnan(collate_graphs([item], require_targets=False)["target_shift"]).all())
        with self.assertRaises(ValueError):
            collate_graphs([item])
        item.update(molecule_id="m", source_molecule_id="m", dataset="CHAMPS_original_hidden",
                    pair_index=np.array([[0, 1]]), coupling_type=np.array([0]),
                    targets_hz=np.zeros((1, 5)), path_nodes=np.array([[[0, 1, -1, -1]]]),
                    path_geometry=np.zeros((1, 1, 6)))
        self.assertTrue(torch.isnan(collate_couplings([item], require_targets=False)["targets_hz"]).all())
        with self.assertRaises(ValueError):
            collate_couplings([item])

    def test_reference_join_uses_atom_identity(self):
        frame = pd.DataFrame(dict(molecule_uid=["a", "a"], atom_index_conformer=[0, 1],
                                  nucleus=["1H", "1H"], target_ppm=[np.nan, np.nan],
                                  prediction_ppm=[1., 2.]))
        reference = frame.iloc[::-1].copy()
        reference["target_ppm"] = [20., 10.]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "reference.csv"
            reference.to_csv(path, index=False)
            joined = attach_references(frame, path, "delta")
            self.assertEqual(joined.target_ppm.tolist(), [10., 20.])
            reference.iloc[:1].to_csv(path, index=False)
            with self.assertRaises(ValueError):
                attach_references(frame, path, "delta")


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
