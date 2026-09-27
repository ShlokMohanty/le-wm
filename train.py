import os
from functools import partial
from pathlib import Path
import torch.nn.functional as F
from einops import rearrange
import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from module import SIGReg
from utils import (
    get_column_normalizer,
    get_img_preprocessor,
    get_rotation_transform,
    rotation_output_dim,
    SaveCkptCallback,
)
ROTATION_COLUMNS={
    "action_rot_quat": "quat",
    "action_yaw": "yaw",
}

def aux_enabled(cfg):
    aux_cfg = cfg.loss.get("aux_ee", None)
    return aux_cfg is not None and (
        aux_cfg.get("weight_trans", 0) > 0
        or aux_cfg.get("weight_rot", 0) > 0
        or aux_cfg.get("weight_grip", 0) > 0
    )

def build_aux_action_slices(cfg, dataset, action_dim):
    component_slices = {}
    offset = 0
    for col in cfg.data.dataset.keys_to_load:
        if not col.startswith("action_"):
            continue
        col_dim = rotation_output_dim(ROTATION_COLUMNS[col]) if col in ROTATION_COLUMNS else dataset.get_dim(col)
        if "grip" in col:
            component_slices["grip"] = [offset, offset + col_dim]
        elif col in ROTATION_COLUMNS or "rot" in col or "yaw" in col:
            component_slices["rot"] = [offset, offset + col_dim]
        elif "trans" in col or "pos" in col:
            component_slices["trans"] = [offset, offset + col_dim]
        offset += col_dim

    if component_slices:
        required = {"trans", "rot", "grip"}
        missing = required.difference(component_slices)
        if missing:
            raise ValueError(
                f"partial aux action mapping from keys_to_load; missing {sorted(missing)}"
            )
        return component_slices

    trans_dim = int(cfg.model.aux_decoder.trans_dim)
    rot_dim = int(cfg.model.aux_decoder.rot_dim)
    grip_dim = int(cfg.model.aux_decoder.grip_dim)
    if trans_dim + rot_dim + grip_dim > action_dim:
        raise ValueError(
            f"Aux action dims ({trans_dim}+{rot_dim}+{grip_dim}) exceed action dim ({action_dim})"
        )

    return {
        "trans": [0, trans_dim],
        "rot": [trans_dim, trans_dim + rot_dim],
        "grip": [trans_dim + rot_dim, trans_dim + rot_dim + grip_dim],
    }

def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  
    if aux_enabled(cfg):
        aux_pred = self.model.decode_aux(pred_emb)
        gt_action = batch["action"][:, cfg.history_size:]
        slices = cfg.model.aux_action_slices
        trans_gt = gt_action[..., slices["trans"][0]:slices["trans"][1]]
        rot_gt = gt_action[..., slices["rot"][0]:slices["rot"][1]]
        grip_gt = gt_action[..., slices["grip"][0]:slices["grip"][1]]
        aux_trans_loss = F.mse_loss(aux_pred["trans_pred"], rearrange(trans_gt, "b t d -> (b t) d"))
        aux_rot_loss = F.mse_loss(aux_pred["rot_pred"], rearrange(rot_gt, "b t d -> (b t) d"))
        aux_grip_loss = F.mse_loss(aux_pred["grip_pred"], rearrange(grip_gt, "b t d -> (b t) d"))
        output["aux_trans_loss"] = aux_trans_loss
        output["aux_rot_loss"] = aux_rot_loss
        output["aux_grip_loss"] = aux_grip_loss
        output["loss"] = (output["pred_loss"]
                         + lambd * output["sigreg_loss"]
                         + cfg.loss.aux_ee.weight_trans * aux_trans_loss
                         + cfg.loss.aux_ee.weight_rot * aux_rot_loss
                         + cfg.loss.aux_ee.weight_grip * aux_grip_loss
                         )

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True) #converting the tuples lists and dictioneries from omegaconf to python dependencies for support
    dataset_name = dataset_cfg.pop("name") #name is popped out 
    cache_dir = os.environ.get("LOCAL_DATASET_DIR", None) #cache directory is used as the local dataset dir
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )# stabel world model dataset loading dataset_name, transform is none cache-dir 
    transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
    
    with open_dict(cfg):
        rotation_dim_delta = 0
        rotation_kind = None
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            if col in ROTATION_COLUMNS:
                kind = ROTATION_COLUMNS[col]
                if rotation_kind is not None and rotation_kind != kind:
                    raise ValueError(f"multiple rotation representations configured: {rotation_kind}, {kind}")
                rotation_kind = kind
                raw_dim = dataset.get_dim(col)
                new_dim = rotation_output_dim(kind)
                rotation_dim_delta += (new_dim - raw_dim)
                transforms.append(get_rotation_transform(col, col, kind))
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        action_dim = dataset.get_dim("action") + rotation_dim_delta
        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * action_dim
        )
        if rotation_kind is not None:
            cfg.model.aux_decoder.rot_dim = rotation_output_dim(rotation_kind)
        cfg.model.aux_action_slices = build_aux_action_slices(cfg, dataset, action_dim)

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
    )

    train = torch.utils.data.DataLoader(train_set, **cfg.loader,shuffle=True, drop_last=True, generator=rnd_gen)
    val = torch.utils.data.DataLoader(val_set, **cfg.loader, shuffle=False, drop_last=False)
    
    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model) #world model hydra.utils.instantiate cfg model

    optimizers = {
        'model_opt': {
            "modules": 'model',
            "optimizer": dict(cfg.optimizer),
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        },
    }

    data_module = spt.data.DataModule(train=train, val=val) #what is the difference between the data_module and the world_model
    world_model = spt.Module(
        model = world_model,
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg), #partial will call the function lejepa forward and will keep the argument fixed for example the cfg 
        optim=optimizers, # i want to know when is the optimizer is passsed as a dictionary or what ?
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get("subdir") or ""
    run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id) #what is the purpose of getting the cache_dir and the run_id 

    logger = None #why is a logger used in general ?
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        OmegaConf.save(cfg, f)

    object_dump_callback = SaveCkptCallback(
        run_name=cfg.output_model_name, cfg=cfg.model, epoch_interval=1,
    ) # how is the saveCkptCallback maintained or saved ?

    trainer = pl.Trainer( #lighting trainer property s being used
        **cfg.trainer,
        callbacks=[object_dump_callback],
        num_sanity_val_steps=1, #sanity validation steps is 1 
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    return


if __name__ == "__main__":
    run()
