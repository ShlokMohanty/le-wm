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
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback


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
    if cfg.loss.get("aux_ee", {}).get("weight", 0) > 0:
        aux_pred = self.model.decode_aux(pred_emb)
        gt_action = batch["action"][:, cfg.history_size:] 
        trans_gt = gt_action[..., 0:3]
        rot_gt = gt_action[..., 3:6]
        grip_gt = gt_action[..., 6:7]
        aux_trans_loss = F.mse_loss(aux_pred["trans_pred"], rearrange(trans_gt, "b t d -> (b t) d"))
        aux_rot_loss = F.mse_loss(aux_pred["rot_pred"], rearrange(rot_gt, "b t d -> (b t) d"))
        aux_grip_loss = F.mse_loss(aux_pred["grip_pred"], rearrange(grip_gt, "b t d -> (b t) d"))
        output["aux_rot_loss"] = aux_rot_loss
        output["loss"] = (output["pred_loss"]
                         + lambda * output["sigreg_loss"]
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
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

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
