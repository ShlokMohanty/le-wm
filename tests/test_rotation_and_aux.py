import importlib
import sys
import types

import pytest

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - depends on environment
    torch = None


def _install_import_stubs():
    if "hydra" not in sys.modules:
        hydra = types.ModuleType("hydra")
        hydra.main = lambda *args, **kwargs: (lambda fn: fn)
        hydra.utils = types.SimpleNamespace(instantiate=lambda cfg: cfg)
        sys.modules["hydra"] = hydra

    if "lightning" not in sys.modules:
        lightning = types.ModuleType("lightning")
        lightning.Trainer = object
        pytorch = types.ModuleType("lightning.pytorch")
        callbacks = types.ModuleType("lightning.pytorch.callbacks")
        callbacks.Callback = object
        loggers = types.ModuleType("lightning.pytorch.loggers")
        loggers.WandbLogger = object
        sys.modules["lightning"] = lightning
        sys.modules["lightning.pytorch"] = pytorch
        sys.modules["lightning.pytorch.callbacks"] = callbacks
        sys.modules["lightning.pytorch.loggers"] = loggers

    if "stable_pretraining" not in sys.modules:
        spt = types.ModuleType("stable_pretraining")
        transforms = types.SimpleNamespace(
            ToImage=lambda **kwargs: ("to_image", kwargs),
            Resize=lambda *args, **kwargs: ("resize", args, kwargs),
            WrapTorchTransform=lambda fn, source, target: ("wrap", fn, source, target),
            Compose=lambda *ops: ("compose", ops),
        )
        data = types.SimpleNamespace(
            transforms=transforms,
            dataset_stats=types.SimpleNamespace(ImageNet={}),
            random_split=lambda dataset, lengths, generator: (dataset, dataset),
            DataModule=object,
        )
        spt.data = data
        spt.Module = object
        spt.Manager = object
        sys.modules["stable_pretraining"] = spt

    if "stable_worldmodel" not in sys.modules:
        swm = types.ModuleType("stable_worldmodel")
        swm.data = types.SimpleNamespace(
            load_dataset=lambda *args, **kwargs: None,
            utils=types.SimpleNamespace(get_cache_dir=lambda sub_folder=None: "/tmp"),
        )
        sys.modules["stable_worldmodel"] = swm

_install_import_stubs()


def test_import_train_and_utils():
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    train = importlib.import_module("train")
    utils = importlib.import_module("utils")
    assert hasattr(train, "run")
    assert hasattr(utils, "QuaternionTo6D")


def test_quaternion_to_6d_output_dim():
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    from utils import QuaternionTo6D

    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = QuaternionTo6D()(q)
    assert out.shape[-1] == 6
    assert torch.allclose(out[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), atol=1e-6)


def test_yaw_to_sincos_output_dim():
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    from utils import YawToSinCos

    yaw = torch.tensor([[0.5], [1.0]])
    out = YawToSinCos()(yaw)
    assert out.shape[-1] == 2
    theta = torch.tensor(0.5)
    assert torch.allclose(out[0], torch.stack([torch.sin(theta), torch.cos(theta)]), atol=1e-6)

    zero_out = YawToSinCos()(torch.tensor([[0.0]]))
    assert torch.allclose(zero_out[0], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_aux_action_decoder_forward_shapes():
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    from module import AuxActionDecoder

    decoder = AuxActionDecoder(input_dim=16, hidden_dim=32, trans_dim=3, rot_dim=6, grip_dim=1)
    emb = torch.randn(5, 16)
    out = decoder(emb)

    assert set(out.keys()) == {"trans_pred", "rot_pred", "grip_pred"}
    assert out["trans_pred"].shape == (5, 3)
    assert out["rot_pred"].shape == (5, 6)
    assert out["grip_pred"].shape == (5, 1)


def test_aux_enabled_and_build_aux_action_slices():
    if torch is None:
        pytest.skip("torch is not installed in this environment")
    train = importlib.import_module("train")

    class DummyDataset:
        def __init__(self, dims):
            self._dims = dims

        def get_dim(self, key):
            return self._dims[key]

    cfg = types.SimpleNamespace(
        loss={"aux_ee": {"weight_trans": 0.1, "weight_rot": 0.0, "weight_grip": 0.0}},
        data=types.SimpleNamespace(dataset=types.SimpleNamespace(keys_to_load=["pixels", "action"])),
        model=types.SimpleNamespace(aux_decoder=types.SimpleNamespace(trans_dim=3, rot_dim=6, grip_dim=1)),
    )
    assert train.aux_enabled(cfg)
    assert not train.aux_enabled(
        types.SimpleNamespace(loss={}, data=cfg.data, model=cfg.model)
    )
    assert not train.aux_enabled(
        types.SimpleNamespace(
            loss={"aux_ee": {"weight_trans": 0.0, "weight_rot": 0.0, "weight_grip": 0.0}},
            data=cfg.data,
            model=cfg.model,
        )
    )
    slices = train.build_aux_action_slices(cfg, DummyDataset({"action": 10}), action_dim=10)
    assert slices == {"trans": [0, 3], "rot": [3, 9], "grip": [9, 10]}

    cfg_partial = types.SimpleNamespace(
        loss=cfg.loss,
        data=types.SimpleNamespace(dataset=types.SimpleNamespace(keys_to_load=["action_rot_quat"])),
        model=cfg.model,
    )
    with pytest.raises(ValueError):
        train.build_aux_action_slices(cfg_partial, DummyDataset({"action_rot_quat": 4}), action_dim=10)

    cfg_mixed = types.SimpleNamespace(
        loss=cfg.loss,
        data=types.SimpleNamespace(dataset=types.SimpleNamespace(keys_to_load=["action", "action_rot_quat"])),
        model=cfg.model,
    )
    mixed_slices = train.build_aux_action_slices(
        cfg_mixed, DummyDataset({"action_rot_quat": 4}), action_dim=10
    )
    assert mixed_slices == {"trans": [0, 3], "rot": [3, 9], "grip": [9, 10]}

    cfg_mixed_tail = types.SimpleNamespace(
        loss=cfg.loss,
        data=types.SimpleNamespace(
            dataset=types.SimpleNamespace(keys_to_load=["action", "action_rot_quat", "action_grip"])
        ),
        model=cfg.model,
    )
    mixed_tail_slices = train.build_aux_action_slices(
        cfg_mixed_tail, DummyDataset({"action_rot_quat": 4, "action_grip": 1}), action_dim=10
    )
    assert mixed_tail_slices == {"trans": [0, 3], "rot": [3, 9], "grip": [9, 10]}

    cfg_mutated = types.SimpleNamespace(
        loss=cfg.loss,
        data=types.SimpleNamespace(dataset=types.SimpleNamespace(keys_to_load=["action"])),
        model=types.SimpleNamespace(aux_decoder=types.SimpleNamespace(trans_dim=3, rot_dim=99, grip_dim=1)),
    )
    mutated_slices = train.build_aux_action_slices(cfg_mutated, DummyDataset({"action": 10}), action_dim=10)
    assert mutated_slices == {"trans": [0, 3], "rot": [3, 9], "grip": [9, 10]}
