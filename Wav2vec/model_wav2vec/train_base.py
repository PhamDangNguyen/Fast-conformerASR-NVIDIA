from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2ForCTC
from transformers import TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from datasets import load_metric
import numpy as np
import os

from model_wav2vec.data import CustomDataset, DataCollatorCTCWithPadding
from trainer import CustomTrainer

def main():

    # save_dir = 'output/wav2vec2-large-xlsr-53'
    save_dir = 'output/wav2vec2-base-nguyenvulebinh/'
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Wav2Vec2CTCTokenizer(
        './vocab/vocab_base.json', 
        unk_token='<unk>', 
        pad_token='<pad>', 
        word_delimiter_token='|'
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


    wer_metric = load_metric('wer')

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {'wer': wer}


    model = Wav2Vec2ForCTC.from_pretrained(
        'nguyenvulebinh/wav2vec2-base-vietnamese-250h',
        # 'models/base', 
        gradient_checkpointing=True, 
        ctc_loss_reduction='mean', 
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=save_dir,
        report_to="none",
        group_by_length=True,
        per_device_train_batch_size=40,
        evaluation_strategy='steps',
        num_train_epochs=5000,
        fp16=True,
        save_steps=1000,
        eval_steps=10000,
        logging_steps=100,
        learning_rate=1e-6,
        weight_decay=0.005,
        warmup_steps=100,
        save_total_limit=10,
        ignore_data_skip=True,
        dataloader_num_workers=24
    )

    trainer = CustomTrainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )


    last_checkpoint = get_last_checkpoint(save_dir)
    if last_checkpoint:
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
