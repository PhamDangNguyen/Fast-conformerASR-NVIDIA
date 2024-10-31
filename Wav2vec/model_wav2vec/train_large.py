import tqdm
import os
import sys
import numpy as np

from datasets import load_dataset, load_metric
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC

from model_wav2vec.data import CustomDataset, DataCollatorCTCWithPadding
from model_wav2vec.trainer import CustomTrainer

# sys.path.append("/home/ndanh/asr-wav2vec/apex")

def main():
    save_dir = 'checkpoints/wav2vec2-large-nguyenvulebinh-25-12/'
    os.makedirs(save_dir, exist_ok=True)

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

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    train_dataset = CustomDataset('train', processor)
    eval_dataset = CustomDataset('val', processor)
    print(len(train_dataset), len(eval_dataset))

    # wer_metric = load_metric('wer')

    # def compute_metrics(pred):
    #     pred_logits = pred.predictions
    #     pred_ids = np.argmax(pred_logits, axis=-1)
    #     pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    #     pred_str = processor.batch_decode(pred_ids)
    #     label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    #     wer = wer_metric.compute(predictions=pred_str, references=label_str)
    #     return {'wer': wer}

    model = Wav2Vec2ForCTC.from_pretrained(
        'nguyenvulebinh/wav2vec2-large-vi-vlsp2020',
        # 'models/base', 
        # gradient_checkpointing=True, 
        ignore_mismatched_sizes=True,
        ctc_loss_reduction='mean',
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    model.config.ctc_zero_infinity = True

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=save_dir,
        report_to="none",
        group_by_length=True,
        per_device_train_batch_size=3,
        per_device_eval_batch_size=8,
        evaluation_strategy='steps',
        num_train_epochs=50000,
        fp16=True,
        save_steps=20000,
        eval_steps=20000,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=100,
        save_total_limit=3,
        ignore_data_skip=True,
        dataloader_num_workers=16,
        gradient_accumulation_steps=2,
    )

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        # compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )

    last_checkpoint = get_last_checkpoint(save_dir)

    if last_checkpoint:
        print("Finetuned")
        print("-----------------------")
        print(f"last_checkpoint: {last_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        train_result = trainer.train()

    # metrics = train_result.metrics
    # metrics["train_samples"] = len(train_dataset)

    # trainer.save_model()

    # trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)
    # trainer.save_state()


if __name__ == '__main__':
    main()
