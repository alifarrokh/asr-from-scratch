import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from load_dataset import load_hf_dataset
from data_collators import MelSpectrogramDataCollator
from deepspeech2_model import DeepSpeech2Config, LightDeepSpeech2
from vocab import Vocab
from tokenizer import Tokenizer


# Hyper-parameters
train_conf = {
    'exp_name': 'deepspeech/test1',
    'epochs': 500,
    'batch_size': 96,
    'accumulate_grad_batches': 1,
    'learning_rate': 1e-3,
    'log_interval': 0.25,
    'early_stopping_patience': 20, # 5 epochs
    'train_dataloader_workers': 4,
    'test_dataloader_workers': 2,
}

# Config
sampling_rate = 16000

# Load the vocab and tokenizer
vocab = Vocab.from_json('deepspeech2_vocab.json')
tokenizer = Tokenizer(vocab)

# Create the config
config = DeepSpeech2Config(
    n_mels=80,
    hidden_units=384,
    tokenizer=tokenizer,

    # Training Parameters
    learning_rate=train_conf['learning_rate']
)

# Load the dataset
torch.manual_seed(48)
train_dataset = load_hf_dataset('train', sampling_rate=sampling_rate, with_features=True)
test_dataset = load_hf_dataset('test', sampling_rate=sampling_rate, with_features=True)

# Tokenize the dataset
tokenize_fn = lambda batch: {'labels': tokenizer.tokenize(batch['sentence'])}
train_dataset = train_dataset.map(tokenize_fn, batched=True, batch_size=32, remove_columns=['sentence'])
test_dataset = test_dataset.map(tokenize_fn, batched=True, batch_size=32, remove_columns=['sentence'])

# Createa dataloaders
data_collator = MelSpectrogramDataCollator()
train_dataloader = DataLoader(train_dataset, batch_size=train_conf['batch_size'], collate_fn=data_collator, shuffle=True, num_workers=train_conf['train_dataloader_workers'])
test_dataloader = DataLoader(test_dataset, batch_size=train_conf['batch_size'], collate_fn=data_collator, num_workers=train_conf['test_dataloader_workers'])

# Callbacks
lr_monitor = LearningRateMonitor(logging_interval='step')
early_stopping = EarlyStopping(monitor="val_cer", mode="min", patience=train_conf['early_stopping_patience'])
checkpoint_callback = ModelCheckpoint(
    save_top_k=1,
    monitor='val_cer',
    mode='min',
    filename='checkpoint-{epoch:02d}-{val_cer:.2f}',
)
callbacks = [checkpoint_callback, lr_monitor, early_stopping]

# Train
steps_per_epoch = int(len(train_dataset) / (train_conf['batch_size'] * train_conf['accumulate_grad_batches']))
log_steps = max(1, int(steps_per_epoch * train_conf['log_interval']))
model = LightDeepSpeech2(config)
trainer = L.Trainer(
    default_root_dir=f'exps/{train_conf["exp_name"]}',
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    max_epochs=train_conf['epochs'],
    accumulate_grad_batches=train_conf['accumulate_grad_batches'],
    log_every_n_steps=log_steps,
    val_check_interval=train_conf['log_interval'],
    enable_model_summary=False,
    callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    num_sanity_val_steps=0,
)
trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=test_dataloader
)
