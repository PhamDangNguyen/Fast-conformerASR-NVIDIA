# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
from dataclasses import dataclass
from pathlib import Path, PosixPath, WindowsPath
from typing import Optional, Union

import lightning_fabric as fl
import pytorch_lightning as pl

from nemo.lightning import io
from nemo.lightning.base import NEMO_MODELS_CACHE
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils import logging
from nemo.utils.app_state import AppState
from nemo.utils.model_utils import uninject_model_parallel_rank

# Dynamically inherit from the correct Path subclass based on the operating system.
if os.name == "nt":
    BasePath = WindowsPath
else:
    BasePath = PosixPath


def _try_restore_tokenizer(model, ckpt_path):
    from nemo.lightning.io import load_context

    try:
        tokenizer = load_context(ckpt_path, "model.tokenizer")
        model.tokenizer = tokenizer
        model.__io__.tokenizer = tokenizer.__io__
    except:
        # Ignore if the ckpt doesn't have a tokenizer.
        pass
    finally:
        return model


@dataclass(kw_only=True)
class AutoResume:
    """Class that handles the logic for setting checkpoint paths and restoring from
    checkpoints in NeMo.

    Attributes:
        restore_config (Optional[RestoreConfig]): Optional config for selectively restoring specific parts like model weights, optimizer states, etc.
            If the config contains a path from HF or another non-NeMo checkpoint format, the checkpoint will be automatically converted to a NeMo compatible format.
            resume_from_folder or the run's log_dir takes precedence over restore_config.
        resume_from_directory (str): Path to the checkpointing directory to restore from.
        resume_from_path (str): Path to a specific checkpoint to restore from.
        adapter_path (str): Path to any adapter checkpoints.
        resume_if_exists (bool): Whether this experiment is resuming from a previous run. If
            True, it sets trainer._checkpoint_connector._ckpt_path so that the trainer should
            auto-resume. exp_manager will move files under log_dir to log_dir/run_{int}.
            Defaults to False.
        resume_past_end (bool): By default, AutoResume throws an error if resume_if_exists is
            True and a checkpoint matching ``*end.ckpt`` indicating a previous training run
            fully completed. Setting resume_past_end=True disables this behavior and loads the
            last checkpoint.
        resume_ignore_no_checkpoint (bool): AutoResume throws an error if resume_if_exists is
            True and no checkpoint could be found. Setting resume_ignore_no_checkpoint=True
            disables this behavior, in which case exp_manager will print a message and
            continue without restoring.
    """

    restore_config: Optional[RestoreConfig] = None
    resume_from_directory: Optional[str] = None
    resume_from_path: Optional[str] = None
    adapter_path: Optional[str] = None
    resume_if_exists: bool = False
    resume_past_end: bool = False
    resume_ignore_no_checkpoint: bool = False

    WEIGHTS_PATH = "weights"

    def get_weights_path(self, path):
        return Path(path) / self.WEIGHTS_PATH

    def setup(self, trainer: Union[pl.Trainer, fl.Fabric], model=None):
        if isinstance(trainer, fl.Fabric):
            raise NotImplementedError("Fabric is not supported yet.")

        trainer_ckpt_path = self.get_trainer_ckpt_path(model)
        if trainer_ckpt_path:
            trainer.ckpt_path = trainer_ckpt_path
            trainer.checkpoint_callback.last_model_path = trainer_ckpt_path
            # Load artifacts
            if getattr(self.restore_config, 'load_artifacts', False):
                if isinstance(trainer_ckpt_path, AdapterPath):
                    # load tokenizer from the base model during peft resume, in case the first peft checkpoint
                    # is deleted before the current peft checkpoint is saved
                    context_path = trainer_ckpt_path.base_model_path / "context"
                    if not context_path.exists():
                        context_path = trainer_ckpt_path.base_model_path
                else:
                    context_path = self.get_context_path(model)
                model = _try_restore_tokenizer(model, context_path)

        elif self.restore_config:
            new_path = self._extract_path(
                model=model,
                path=self.restore_config.path,
                adapter_path=self.restore_config.adapter_path,
            )
            if isinstance(new_path, AdapterPath):
                self.restore_config.path = new_path.base_model_path
                self.restore_config.adapter_path = str(new_path)
            else:
                self.restore_config.path = str(new_path)
            trainer.strategy.restore_config = self.restore_config
            # Load artifacts
            if self.restore_config.load_artifacts:
                context_path = new_path / "context"
                if not context_path.is_dir():
                    context_path = new_path

                _try_restore_tokenizer(model, context_path)

    def _extract_path(
        self, model: Optional[io.ConnectorMixin], path: str, adapter_path: Optional[str] = None
    ) -> BasePath:
        if "://" in path:
            assert path.startswith("nemo://"), "Only NeMo based paths starting with nemo:// are currently supported."
            _, _path = path.split("://")
            new_path = os.path.join(NEMO_MODELS_CACHE, _path)
        else:
            new_path = path

        if adapter_path:

            maybe_weights_path = self.get_weights_path(adapter_path)
            if maybe_weights_path.is_dir():
                adapter_path = maybe_weights_path

            new_path = AdapterPath(Path(adapter_path), base_model_path=new_path)

        if isinstance(new_path, str):
            new_path = Path(new_path)

        return new_path

    def _resume_peft(self, adapter_meta_path, model):
        with open(adapter_meta_path, "r") as f:
            metadata = json.load(f)

        assert self.restore_config, "PEFT resume requires specifying restore_config"
        base_model_path = self._extract_path(model, self.restore_config.path)
        if base_model_path not in [Path(metadata['model_ckpt_path']), Path(metadata['model_ckpt_path']).parent]:
            logging.warning(
                f"⚠️ When trying to resume a PEFT training run, found mismatching values: "
                f"your specified restore_path points to {base_model_path}, "
                f"but the PEFT checkpoint was trained with "
                f"model_ckpt_path={metadata['model_ckpt_path']}"
            )
        return base_model_path

    def _find_trainer_ckpt_path(self) -> Optional[Path]:
        from nemo.utils.exp_manager import NotFoundError, _filter_out_unfinished_checkpoints

        app_state = AppState()
        log_dir = app_state.log_dir

        checkpoint = None

        # Use <log_dir>/checkpoints/ unless `dirpath` is set
        checkpoint_dir = (
            Path(self.resume_from_directory) if self.resume_from_directory else Path(Path(log_dir) / "checkpoints")
        )

        # when using distributed checkpointing, checkpoint_dir is a directory of directories
        # we check for this here
        dist_checkpoints = [d for d in list(checkpoint_dir.glob("*")) if d.is_dir()]
        end_dist_checkpoints = [d for d in dist_checkpoints if d.match("*end")]
        last_dist_checkpoints = [d for d in dist_checkpoints if d.match("*last")]

        end_chkpt_cnt = len(end_dist_checkpoints)
        end_checkpoints = _filter_out_unfinished_checkpoints(end_dist_checkpoints)
        finished_end_chkpt_cnt = len(end_checkpoints)
        if end_chkpt_cnt > 0 and finished_end_chkpt_cnt == 0:
            raise ValueError(
                "End checkpoint is unfinished and cannot be used to resume the training."
                " Please remove the checkpoint manually to avoid unexpected cosequences, such as"
                " restarting from scratch."
            )

        last_chkpt_cnt = len(last_dist_checkpoints)
        last_checkpoints = _filter_out_unfinished_checkpoints(last_dist_checkpoints)
        finished_last_chkpt_cnt = len(last_checkpoints)
        if last_chkpt_cnt > 0 and finished_last_chkpt_cnt == 0:
            raise ValueError(
                "Last checkpoint is unfinished and cannot be used to resume the training."
                " Please remove the checkpoint manually to avoid unexpected cosequences, such as"
                " restarting from scratch. Hint: Iteration number can be added to the checkpoint name pattern"
                " to maximize chance that there is at least one finished last checkpoint to resume from."
            )

        if not checkpoint_dir.exists() or (not len(end_checkpoints) > 0 and not len(last_checkpoints) > 0):
            if self.resume_ignore_no_checkpoint:
                warn = f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. "
                if checkpoint is None:
                    warn += "Training from scratch."
                logging.warning(warn)
            else:
                if self.restore_config:
                    # resume_if_exists is True but run is not resumable. Do not fail and try to do selective restore later instead.
                    return None
                else:
                    raise NotFoundError(
                        f"There were no checkpoints found in checkpoint_dir or no checkpoint folder at checkpoint_dir :{checkpoint_dir}. Cannot resume."
                    )
        elif len(end_checkpoints) > 0:
            if not self.resume_past_end:
                raise ValueError(
                    f"Found {end_checkpoints[0]} indicating that the last training run has already completed."
                )

            if len(end_checkpoints) > 1:
                if "mp_rank" in str(end_checkpoints[0]):
                    checkpoint = end_checkpoints[0]
                else:
                    raise ValueError(f"Multiple checkpoints {end_checkpoints} that matches *end.ckpt.")
        elif len(last_checkpoints) > 1:
            if any([s for s in ["mp_rank", "tp_rank", "fsdp_shard"] if s in str(last_checkpoints[0])]):
                checkpoint = last_checkpoints[0]
                checkpoint = uninject_model_parallel_rank(checkpoint)
            else:
                # Select the checkpoint with the latest modified time
                checkpoint = sorted(last_checkpoints, key=lambda pth: pth.lstat().st_mtime, reverse=True)[0]
                logging.warning(
                    f"Multiple checkpoints {last_checkpoints} matches *last.ckpt. Selecting one with the latest modified time."
                )
        else:
            checkpoint = last_checkpoints[0]

        return checkpoint

    def get_context_path(self, model: Optional[io.ConnectorMixin] = None) -> Optional[Path]:
        checkpoint = None
        app_state = AppState()
        app_state.restore = self.resume_if_exists
        if self.resume_if_exists:
            checkpoint = self._find_trainer_ckpt_path()

        if checkpoint:
            maybe_context_path = Path(checkpoint) / "context"
            if maybe_context_path.is_dir():
                checkpoint = maybe_context_path
        return checkpoint

    def get_trainer_ckpt_path(self, model: Optional[io.ConnectorMixin] = None) -> Optional[Path]:
        if self.resume_from_path:
            maybe_weights_path = self.get_weights_path(self.resume_from_path)
            return maybe_weights_path if maybe_weights_path.is_dir() else self.resume_from_path

        checkpoint = None
        app_state = AppState()
        app_state.restore = self.resume_if_exists
        if self.resume_if_exists:
            checkpoint = self._find_trainer_ckpt_path()

        if checkpoint:
            maybe_weights_path = self.get_weights_path(checkpoint)
            if maybe_weights_path.is_dir():
                checkpoint = maybe_weights_path

        if checkpoint:
            if self.adapter_path:
                return AdapterPath(Path(self.adapter_path), base_model_path=checkpoint)
            else:
                from nemo.lightning.pytorch.callbacks.peft import _ADAPTER_META_FILENAME

                adapter_meta_path = checkpoint / _ADAPTER_META_FILENAME
                if adapter_meta_path.exists():
                    base_model_path = self._resume_peft(adapter_meta_path, model)
                    return AdapterPath(checkpoint, base_model_path=base_model_path)
                else:
                    return Path(checkpoint)

        return None


class AdapterPath(BasePath):
    """Path object for adapter paths which include a field for the base model the adapters are trained on
    to facilitate model loading."""

    base_model_path: Optional[Path]

    def __new__(cls, *args, base_model_path: Optional[Path] = None, **kwargs):
        output = super().__new__(cls, *args, **kwargs)
        output.base_model_path = base_model_path
        return output

    def __repr__(self):
        return "{}({!r}, base_model_path={})".format(self.__class__.__name__, self.as_posix(), self.base_model_path)
