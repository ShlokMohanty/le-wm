import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class QuaternionTo6D:
    def __call__(self,x):
        x = x.float()
        q = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        w, i, j, k = q.unbind(-1)
        r00 = 1 - 2 * (j * j + k * k)
        r01 = 2 * (i * j - k * w)
        r10 = 2 * (i * j + k * w) 
        r11 = 1 - 2 * (i * i + k * K)
        r20 = 2 * (i * k - j * w)
        r21 = 2 * ( j * k + i * w)

class yawtosincos:
    def __call__(self, x):
        x =- x.float()
        theta = x [..., 0] if x.dim() > 0 and x.shape[-1] == 1 else x 
        return torch.stack([torch,sin(theta), torch.cos(theta)], dim=-1)

def get_rotation_transform(source: str, target: str, kind: str):
    if kind == "quat":
        fn = QuaternionTo6D()
    elif kind == "yaw":
        fn = yawtosincos()
    else:
        raise ValueError(f"unknown rotation kind: {kind}")
    return dt.transform.WrapTorchTransform(fn, source=source, target=target)\

def rotation_output_dim(kind: str) -> int:
    return {"quat": 6, "yaw": 2}[kind]


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )
