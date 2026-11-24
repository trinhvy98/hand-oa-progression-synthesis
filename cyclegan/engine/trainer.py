"""Modular CycleGAN trainer with DDP multi-GPU, AMP, and val-based early stopping."""

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from cyclegan.models.cyclegan import CycleGANModel
from hand_oa.data import (
    SingleDomainDataset,
    UnpairedDomainDataset,
    build_generative_transform,
    load_manifest_from_config,
)
from hand_oa.evaluation import (
    InceptionFeatureExtractor,
    evaluate_translation_validation,
    extract_features_from_loader,
)
from hand_oa.utils import (
    barrier_if_distributed,
    broadcast_bool,
    ensure_dir,
    is_main_process,
    reduce_mean_dict,
    resolve_project_path,
    save_json,
    seed_everything,
)


def build_cyclegan_from_config(
    cfg: dict[str, Any],
    device: torch.device | None = None,
    is_distributed: bool = False,
    local_rank: int = 0,
) -> CycleGANModel:
    """Instantiate CycleGANModel from parsed cyclegan.yaml configuration."""
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    return CycleGANModel(
        in_channels=int(data_cfg["in_channels"]),
        ngf=int(model_cfg["ngf"]),
        ndf=int(model_cfg["ndf"]),
        n_res_blocks=int(model_cfg["n_res_blocks"]),
        gan_mode=str(model_cfg["gan_mode"]),
        pool_size=int(model_cfg["pool_size"]),
        lr=float(train_cfg["lr"]),
        beta1=float(train_cfg["beta1"]),
        beta2=float(train_cfg["beta2"]),
        lambda_cycle_a=float(train_cfg["lambda_cycle_a"]),
        lambda_cycle_b=float(train_cfg["lambda_cycle_b"]),
        lambda_identity=float(train_cfg.get("lambda_identity", 0.0)),
        epochs=int(train_cfg["epochs"]),
        decay_start_epoch=int(train_cfg["decay_start_epoch"]),
        use_amp=bool(train_cfg.get("use_amp", False)),
        is_distributed=is_distributed,
        local_rank=local_rank,
        device=device,
    )


class CycleGANTrainer:
    """Manages CycleGAN training, validation scoring, early stopping, and checkpoints."""

    def __init__(
        self,
        data_cfg: dict[str, Any],
        cyclegan_cfg: dict[str, Any],
        project_root: str | Path,
        device: torch.device | None = None,
        is_distributed: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.data_cfg = data_cfg
        self.cfg = cyclegan_cfg
        self.project_root = Path(project_root)
        self.is_distributed = is_distributed
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

        seed = int(cyclegan_cfg["experiment"].get("seed", 42)) + rank
        seed_everything(seed)

        self.output_dir = resolve_project_path(
            cyclegan_cfg["experiment"]["output_dir"], self.project_root
        )
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.samples_dir = self.output_dir / "samples"
        if is_main_process(self.rank):
            ensure_dir(self.output_dir)
            ensure_dir(self.ckpt_dir)
            ensure_dir(self.samples_dir)

        dataset_cfg = data_cfg["dataset"]
        if is_main_process(self.rank):
            self.manifest, self.images_dir = load_manifest_from_config(data_cfg, self.project_root)
        barrier_if_distributed(self.is_distributed)
        if not is_main_process(self.rank):
            self.manifest, self.images_dir = load_manifest_from_config(data_cfg, self.project_root)

        cg_data = cyclegan_cfg["data"]
        joints = cg_data.get("joints", dataset_cfg["joints"])
        train_transform = build_generative_transform(
            image_size=int(cg_data["image_size"]),
            in_channels=int(cg_data["in_channels"]),
            is_train=True,
        )
        val_transform = build_generative_transform(
            image_size=int(cg_data["image_size"]),
            in_channels=int(cg_data["in_channels"]),
            is_train=False,
        )

        # Explicitly use split='train' only; test split is completely untouched
        self.train_dataset = UnpairedDomainDataset(
            manifest_df=self.manifest,
            images_dir=self.images_dir,
            split="train",
            domain_a_grades=cg_data["domain_a_grades"],
            domain_b_grades=cg_data["domain_b_grades"],
            joints=joints,
            transform=train_transform,
            is_train=True,
        )
        self.val_dataset = UnpairedDomainDataset(
            manifest_df=self.manifest,
            images_dir=self.images_dir,
            split="val",
            domain_a_grades=cg_data["domain_a_grades"],
            domain_b_grades=cg_data["domain_b_grades"],
            joints=joints,
            transform=val_transform,
            is_train=False,
        )

        global_batch_size = int(cg_data["batch_size"])
        local_batch_size = max(1, global_batch_size // max(1, world_size))
        configured_workers = int(cg_data["num_workers"])
        num_workers = (
            max(1, configured_workers // max(1, world_size))
            if (is_distributed and configured_workers > 0)
            else configured_workers
        )
        pin_mem = torch.cuda.is_available()

        self.train_sampler = (
            DistributedSampler(
                self.train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=False,
            )
            if is_distributed
            else None
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=local_batch_size,
            shuffle=(self.train_sampler is None),
            sampler=self.train_sampler,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=min(4, len(self.val_dataset)),
            shuffle=False,
            num_workers=0,
        )
        self.model = build_cyclegan_from_config(
            cyclegan_cfg,
            device=device,
            is_distributed=is_distributed,
            local_rank=local_rank,
        )

        self.es_cfg = dict(cyclegan_cfg.get("early_stopping", {}))
        self.val_source_loader: DataLoader | None = None
        self.val_target_feats = None
        self.inception_extractor: InceptionFeatureExtractor | None = None

        if self.es_cfg.get("enabled", False) and is_main_process(self.rank):
            val_src_ds = SingleDomainDataset(
                manifest_df=self.manifest,
                images_dir=self.images_dir,
                split="val",
                grades=cg_data["domain_a_grades"],
                joints=joints,
                transform=val_transform,
            )
            val_tgt_ds = SingleDomainDataset(
                manifest_df=self.manifest,
                images_dir=self.images_dir,
                split="val",
                grades=cg_data["domain_b_grades"],
                joints=joints,
                transform=val_transform,
            )
            eval_bs = max(8, local_batch_size)
            self.val_source_loader = DataLoader(val_src_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
            val_target_loader = DataLoader(val_tgt_ds, batch_size=eval_bs, shuffle=False, num_workers=0)
            self.inception_extractor = InceptionFeatureExtractor(
                device=self.model.device,
                pretrained=bool(self.es_cfg.get("use_pretrained_inception", True)),
            )
            self.val_target_feats = extract_features_from_loader(
                self.inception_extractor, val_target_loader, image_key="image"
            )

    def _save_validation_grid(self, epoch: int) -> Path:
        batch = next(iter(self.val_loader))
        preview = self.model.preview_cycles(batch["A"], batch["B"])
        stacked = torch.cat(
            [
                preview["real_a"],
                preview["fake_b"],
                preview["rec_a"],
                preview["real_b"],
                preview["fake_a"],
                preview["rec_b"],
            ],
            dim=0,
        )
        grid = make_grid(stacked, nrow=batch["A"].shape[0], normalize=True, value_range=(-1.0, 1.0))
        out_path = self.samples_dir / f"epoch_{epoch:04d}.png"
        save_image(grid, out_path)
        return out_path

    def _evaluate_validation(self, epoch: int) -> dict[str, Any]:
        assert self.val_source_loader is not None
        assert self.val_target_feats is not None
        assert self.inception_extractor is not None
        metrics = evaluate_translation_validation(
            forward_fn=lambda x: self.model.translate(x, direction="a2b"),
            backward_fn=lambda x: self.model.translate(x, direction="b2a"),
            val_source_loader=self.val_source_loader,
            real_target_feats=self.val_target_feats,
            extractor=self.inception_extractor,
            device=self.model.device,
        )
        return {"epoch": epoch, **metrics}

    def fit(
        self,
        max_epochs: int | None = None,
        max_steps_per_epoch: int | None = None,
        resume_checkpoint: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Run CycleGAN training loop with periodic validation scoring and early stopping."""
        train_cfg = self.cfg["training"]
        epochs = int(max_epochs if max_epochs is not None else train_cfg["epochs"])
        save_every = int(train_cfg["save_every_epochs"])
        sample_every = int(train_cfg["sample_every_epochs"])

        es_enabled = bool(self.es_cfg.get("enabled", False))
        eval_every = int(self.es_cfg.get("eval_every_epochs", save_every))
        min_epochs = int(self.es_cfg.get("min_epochs", 40))
        patience = int(self.es_cfg.get("patience", 8))
        min_delta = float(self.es_cfg.get("min_delta", 0.0005))
        min_cycle_ssim = float(self.es_cfg.get("min_cycle_ssim", 0.75))
        min_delta_l1 = float(self.es_cfg.get("min_delta_l1", 0.015))

        start_epoch = 1
        history: list[dict[str, Any]] = []
        val_history: list[dict[str, Any]] = []
        history_csv = self.output_dir / "loss_history.csv"
        val_csv = self.output_dir / "val_metrics.csv"

        best_kid: float | None = None
        best_epoch: int | None = None
        no_improve_evals = 0

        if resume_checkpoint is not None:
            ckpt_path = resolve_project_path(resume_checkpoint, self.project_root)
            loaded_epoch = self.model.load_checkpoint(ckpt_path, load_optimizers=True)
            start_epoch = loaded_epoch + 1
            if is_main_process(self.rank):
                if history_csv.is_file():
                    prev_df = pd.read_csv(history_csv)
                    history = prev_df[prev_df["epoch"] <= loaded_epoch].to_dict("records")
                if val_csv.is_file():
                    prev_val = pd.read_csv(val_csv)
                    prev_val = prev_val[prev_val["epoch"] <= loaded_epoch]
                    val_history = prev_val.to_dict("records")
                    qualified_rows = prev_val[prev_val.get("qualified", True) == True]  # noqa: E712
                    if not qualified_rows.empty:
                        best_idx = qualified_rows["val_kid"].idxmin()
                        best_kid = float(qualified_rows.loc[best_idx, "val_kid"])
                        best_epoch = int(qualified_rows.loc[best_idx, "epoch"])
            barrier_if_distributed(self.is_distributed)

        for epoch in range(start_epoch, epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            epoch_sums: dict[str, float] = {}
            steps = 0
            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
                disable=not is_main_process(self.rank),
            )
            for step_idx, batch in enumerate(pbar):
                if max_steps_per_epoch is not None and step_idx >= max_steps_per_epoch:
                    break
                losses = self.model.train_step(batch["A"], batch["B"])
                for key, val in losses.items():
                    epoch_sums[key] = epoch_sums.get(key, 0.0) + val
                steps += 1
                pbar.set_postfix(g=f"{losses['loss_g']:.3f}", da=f"{losses['loss_d_a']:.3f}")

            local_avg = {k: v / max(1, steps) for k, v in epoch_sums.items()}
            avg_losses = reduce_mean_dict(
                local_avg,
                is_distributed=self.is_distributed,
                world_size=self.world_size,
                device=self.model.device,
            )
            lr_now = self.model.step_schedulers()
            record = {"epoch": epoch, "lr": lr_now, **avg_losses}
            history.append(record)

            should_stop = False
            if is_main_process(self.rank):
                if epoch % sample_every == 0 or epoch == epochs:
                    self._save_validation_grid(epoch)
                if epoch % save_every == 0 or epoch == epochs:
                    self.model.save_checkpoint(
                        self.ckpt_dir / f"epoch_{epoch:04d}.pt",
                        epoch=epoch,
                        extra={"losses": avg_losses},
                    )
                pd.DataFrame(history).to_csv(history_csv, index=False)

                if es_enabled and (epoch % eval_every == 0 or epoch == epochs):
                    val_rec = self._evaluate_validation(epoch)
                    qualified = (
                        val_rec["val_cycle_ssim"] >= min_cycle_ssim
                        and val_rec["val_delta_l1"] >= min_delta_l1
                    )
                    val_rec["qualified"] = bool(qualified)
                    val_history.append(val_rec)
                    pd.DataFrame(val_history).to_csv(val_csv, index=False)

                    improved = qualified and (
                        best_kid is None or val_rec["val_kid"] < best_kid - min_delta
                    )
                    if improved:
                        best_kid = float(val_rec["val_kid"])
                        best_epoch = epoch
                        no_improve_evals = 0
                        self.model.save_checkpoint(
                            self.ckpt_dir / "best_checkpoint.pt",
                            epoch=epoch,
                            extra={"losses": avg_losses, "val_metrics": val_rec},
                        )
                        save_json(
                            {"best_epoch": best_epoch, "best_val_kid": best_kid, **val_rec},
                            self.output_dir / "best_checkpoint_info.json",
                        )
                    elif epoch >= min_epochs:
                        no_improve_evals += 1

                    print(
                        f"[Val Epoch {epoch}] KID={val_rec['val_kid']:.5f} | "
                        f"FID={val_rec['val_fid']:.2f} | "
                        f"CycleSSIM={val_rec['val_cycle_ssim']:.4f} | "
                        f"DeltaL1={val_rec['val_delta_l1']:.4f} | "
                        f"Qualified={qualified} | BestEpoch={best_epoch} | "
                        f"Patience={no_improve_evals}/{patience}"
                    )

                    if epoch >= min_epochs and no_improve_evals >= patience:
                        should_stop = True
                        print(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(best_epoch={best_epoch}, best_val_kid={best_kid:.5f})."
                        )

            should_stop = broadcast_bool(should_stop, self.is_distributed, self.model.device)
            barrier_if_distributed(self.is_distributed)
            if should_stop:
                break

        return history
