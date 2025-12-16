import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable CPU fallback for MPS
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable high watermark for MPS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from pathlib import Path
import logging

import random
import pandas as pd
import math
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from config import Configurator
from data import NeuralDataset, collate_transducer, download_data, get_dataframes
from training import Trainer, EarlyStopping
from losses import RNNTLoss
from models import MyEncoder, MyPredictor, MyJoiner, RNNT, predict_on_sample

Config = Configurator()

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def main():
    # ------------------------ Load data from hdf5 files to Dataframe ------------------------
    path = Config.DATA_PATH
    device = Config.DEVICE
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Using device: {device}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info(f"Model path: {Config.MODEL_PATH}")
    logger.info(f"Training mode: {Config.TRAIN}")

    # Download dataset if not present
    if not os.path.exists(path):
        download_data()

    train_df, val_df = get_dataframes(path, debug=Config.DEBUG)
    # Collect all unique ASCII values from training data
    unique_ascii_ids = set()
    for seq in train_df['transcriptions']:
        # seq might be numpy array
        for char_code in seq:
            if char_code != 0:  # Ignore 0/Blank during collection
                unique_ascii_ids.add(int(char_code))

    unique_list = sorted(list(unique_ascii_ids))

    # Configure the Dense Mapping
    Config.set_vocabulary(unique_list)

    # ------------------------ Dataset and Dataloader ------------------------
    train_dataset = NeuralDataset(train_df, char_to_id=Config.CHAR_TO_ID, augment=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_transducer(b, batch_first=Config.BATCH_FIRST)
    )

    val_dataset = NeuralDataset(val_df, char_to_id=Config.CHAR_TO_ID, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_transducer(b, batch_first=Config.BATCH_FIRST)
    )

    # ------------------------ Define Model ------------------------
    if Config.FEATURE_LEN is None or Config.VOCAB_SIZE is None or Config.VOCAB_SIZE is None or Config.VOCABULARY is None:
        raise ValueError("Vocabulary or feature length not set properly in Config.")

    encoder = MyEncoder(input_dim=Config.FEATURE_LEN, hidden_dim=256, layers=2, dropout=0.2, conformer=False)
    predictor = MyPredictor(vocab_size=Config.VOCAB_SIZE, hidden_dim=256, blank_id=Config.BLANK_ID)
    joiner = MyJoiner(input_dim=256, output_dim=Config.VOCAB_SIZE)
    model = RNNT(encoder, predictor, joiner).to(device)

    loss = RNNTLoss(blank_id=Config.BLANK_ID)
    # loss = FastRNNTLoss(blank_id=Config.BLANK_ID)

    model_checkpoint_path = str(Path(Config.MODEL_PATH) / f"{model}_{loss}_{Config.MODEL_STR}")

    # if not Config.TRAIN:
    #     try:
    #         model.load_state_dict(torch.load(model_checkpoint_path, map_location=device, weights_only=True))
    #         print(f"Loaded model from {model_checkpoint_path}")
    #     except Exception as e:
    #         print(f"Could not load model: {e}")

    # number of model parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {num_params:,} total parameters.")
    print(f"Model has {num_trainable_params:,} trainable parameters.")
    print(model.describe())

    # ------------------------ Training ------------------------
    num_epochs = 100
    lr = 1e-3

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-2
    )

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=lr,
    #     steps_per_epoch=len(train_loader),
    #     epochs=num_epochs,
    #     anneal_strategy='cos'
    # )
    lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 1000, len(train_loader) * num_epochs)

     # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        loss_fn=loss,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=num_epochs,
        early_stop=EarlyStopping(patience=10, min_delta=1e-3, path=model_checkpoint_path),
        batch_first=Config.BATCH_FIRST,
    )

    if Config.TRAIN:
        print("Starting training...")
        trainer.run()

    # Predict on one sample
    rnd_i = random.randint(0, len(val_dataset) - 1)
    x_sample, y_sample = val_dataset[rnd_i]
    y_sample = y_sample.cpu().numpy()
    y_sample = ''.join([Config.ID_TO_CHAR[idx] for idx in y_sample if idx != Config.BLANK_ID])

    predicted_text = predict_on_sample(
        model,
        x_sample,
        vocab_map=Config.ID_TO_CHAR,
        device=device
    )

    print(f"Predicted text on one sample: {predicted_text}")
    print(f"Ground truth text: {y_sample}")


    # ------------------------ Evaluation on Test Set ------------------------
    # avg_cer, avg_wer, results = run_evaluate_text(model, val_loader, blank_id=Config.BLANK_ID, device=device)
    # df = pd.DataFrame(results)
    # df.to_csv("../text_evaluation_results.csv", index=False)
    #
    # print(f"Validation CER: {avg_cer:.3f}%, WER: {avg_wer:.3f}%")

if __name__ == '__main__':
    main()