import time
from transformers.trainer import (
    SequentialDistributedSampler, 
    SequentialSampler,
    # DistributedSamplerWithLoop,
)
from transformers import Trainer
from transformers.trainer_pt_utils import find_batch_size, IterableDatasetShard, nested_numpify
from transformers.trainer_utils import has_length, denumpify_detensorize, speed_metrics

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
from typing import Any, Dict, List, Optional, Union
import collections
import torch
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
from datasets import load_dataset, load_metric
import math
import sys
import os

tokenizer = Wav2Vec2CTCTokenizer(
        os.path.join(os.path.dirname(__file__), 'vocab/vocab_large.json'),
        unk_token='<unk>',
        pad_token='<pad>',
        word_delimiter_token=' '
    ) 
feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=False
)

    # feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("wav2vec2_pretraining/saved_model/epoch_4")

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


wer_metric = load_metric('wer')
def compute_metrics(pred_logits,label_ids):
    pred_ids = np.argmax(pred_logits, axis=-1)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return wer

class CustomTrainer(Trainer):

    def _get_train_sampler(self):
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
            self.train_dataset, collections.abc.Sized
        ):
            return None 
        
        if self.args.world_size <= 1:
            return RandomSampler(self.train_dataset)
        # elif self.args.parallel_mode == ParallelMode.TPU and not self.args.dataloader_drop_last:
        #     # Use a loop for TPUs when drop_last is False to have all batches have the same size.
        #     return DistributedSamplerWithLoop(
        #         self.train_dataset,
        #         batch_size=self.args.per_device_train_batch_size,
        #         num_replicas=self.args.world_size,
        #         rank=self.args.process_index,
        #     )
        else:
            return DistributedSampler(
                self.train_dataset, num_replicas=self.args.world_size, rank=self.args.process_index
            )
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    
    def _get_eval_sampler(self, eval_dataset):
        if self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)


    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        model = self._wrap_model(self.model, training=False)
        model.eval()
        batch_size = self.args.per_device_eval_batch_size

        args = self.args

        prediction_loss_only = args.prediction_loss_only

        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        eval_dataset = getattr(eval_dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None
        
        # losses/preds/labels on CPU (final containers)
        all_losses = []
        all_wer = []
        
        observed_num_examples = 0
        for step, inputs in enumerate(eval_dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            if loss > 10000:
                print(f"loss: {loss}")
                sys.exit()
            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses = nested_numpify(losses)
                all_losses.extend(losses)
                
            
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels = nested_numpify(labels)

            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = nested_numpify(logits)

            

            wer = compute_metrics(logits, labels)
            all_wer.append(wer)
            
            
            
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(eval_dataloader):
                num_samples = self.num_examples(eval_dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        
        all_losses = all_losses[:num_samples]
        all_losses = torch.tensor(all_losses,dtype=torch.float32)
        torch.save(all_losses, 'all_losses.pt')
        metrics = {'wer': np.mean(np.array(all_wer))}
        metrics = denumpify_detensorize(metrics)
        metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=num_samples,
                num_steps=math.ceil(num_samples / total_batch_size),
            )
        )

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics