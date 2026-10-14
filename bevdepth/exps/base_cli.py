# Copyright (c) Megvii Inc. All rights reserved.
import os
import inspect
from argparse import ArgumentParser

import pytorch_lightning as pl

from bevdepth.callbacks.ema import EMACallback
from bevdepth.utils.torch_dist import all_gather_object, synchronize

from .nuscenes.base_exp import BEVDepthLightningModel


def run_cli(model_class=BEVDepthLightningModel,
            exp_name='base_exp',
            use_ema=False,
            extra_trainer_config_args=None):

    extra_trainer_config_args = extra_trainer_config_args or {}

    parser = ArgumentParser(add_help=True)

    # BEVDepth-specific arguments
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-p', '--predict', action='store_true')
    parser.add_argument('-b', '--batch_size_per_device', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt_path', type=str)

    # Common Lightning Trainer arguments
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--accelerator', type=str, default=None)
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--max_epochs', type=int,
                        default=extra_trainer_config_args.get('epochs', 24))
    parser.add_argument('--limit_train_batches', type=float, default=1.0)
    parser.add_argument('--limit_val_batches', type=float, default=0.0)
    parser.add_argument('--num_sanity_val_steps', type=int, default=0)
    parser.add_argument('--gradient_clip_val', type=float, default=5.0)
    parser.add_argument('--precision', type=str, default='16-mixed')
    parser.add_argument('--profiler', type=str, default='simple')
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--default_root_dir', type=str,
                        default=os.path.join('./outputs/', exp_name))
    parser.add_argument('--enable_checkpointing', action='store_true',
                        default=True)
    parser.add_argument('--amp_backend', type=str, default='native')

    parser = model_class.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))

    # Translate legacy --gpus to Lightning 2.x configuration
    devices = args.devices if args.devices is not None else args.gpus
    accelerator = args.accelerator or ('gpu' if devices > 0 else 'cpu')

    trainer_kwargs = dict(
        accelerator=accelerator,
        devices=devices if accelerator == 'gpu' else 1,
        strategy=args.strategy,
        max_epochs=args.max_epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=args.num_sanity_val_steps,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision if accelerator == 'gpu' else '32-true',
        profiler=args.profiler,
        deterministic=args.deterministic,
        default_root_dir=args.default_root_dir,
        enable_checkpointing=args.enable_checkpointing,
    )

    callbacks = []
    if use_ema:
        train_dataloader = model.train_dataloader()
        ema_callback = EMACallback(
            len(train_dataloader.dataset) * args.max_epochs)
        callbacks.append(ema_callback)

    if callbacks:
        trainer_kwargs['callbacks'] = callbacks

    trainer_kwargs["log_every_n_steps"] = 1
    trainer = pl.Trainer(**trainer_kwargs)

    if args.evaluate:
        trainer.test(model, ckpt_path=args.ckpt_path)

    elif args.predict:
        predict_step_outputs = trainer.predict(
            model, ckpt_path=args.ckpt_path)

        all_pred_results = []
        all_img_metas = []

        for predict_step_output in predict_step_outputs:
            for i in range(len(predict_step_output)):
                all_pred_results.append(predict_step_output[i][:3])
                all_img_metas.append(predict_step_output[i][3])

        synchronize()

        len_dataset = len(model.test_dataloader().dataset)
        all_pred_results = sum(
            map(list, zip(*all_gather_object(all_pred_results))),
            [])[:len_dataset]

        all_img_metas = sum(
            map(list, zip(*all_gather_object(all_img_metas))),
            [])[:len_dataset]

        model.evaluator._format_bbox(
            all_pred_results,
            all_img_metas,
            os.path.dirname(args.ckpt_path)
        )

    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)
