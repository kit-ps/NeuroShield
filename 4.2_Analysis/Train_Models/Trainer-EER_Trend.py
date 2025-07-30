#python3 Trainer-EER_Trend.py --chosen_loss SupConLoss --cuda_devices 0 --train Emotive > out11.log 2>&1 &
#python3 Trainer-EER_Trend.py --chosen_loss SupConLoss --cuda_devices 1 --train DSIVR300 > out12.log 2>&1 &
#python3 Trainer-EER_Trend.py --chosen_loss SupConLoss --cuda_devices 2 --train Muse > out13.log 2>&1 &
#python3 Trainer-EER_Trend.py --chosen_loss SupConLoss --cuda_devices 3 --train All > out14.log 2>&1 &


import os
import random
import json
import argparse  
import sqlite3
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, RMSprop, AdamW  
from pytorch_metric_learning import losses, miners
import torch.nn as nn
import torch.nn.functional as F
import selfeeg
import optuna
from optuna.exceptions import TrialPruned
from torch.amp import autocast, GradScaler
from optuna.storages import RDBStorage
from torch.utils.tensorboard import SummaryWriter
from pytorch_metric_learning.samplers import MPerClassSampler
from torch.utils.data import Subset
from typing import Tuple
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from collections import defaultdict
from pyeer.eer_info import get_eer_stats
import random




# fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# parse command-line arguments
parser = argparse.ArgumentParser(description="Train EEG models with metric learning losses and configurable GPUs")
parser.add_argument('--chosen_loss', type=str, default='SupConLoss',
                    choices=['ArcFaceLoss','TripletMarginLoss','SupConLoss','LiftedStructureLoss','SoftTripleLoss'])
parser.add_argument('--cuda_devices', type=str, default='0',
                    help='Comma-separated list of CUDA_VISIBLE_DEVICES')  
parser.add_argument('--train', type=str, default='train')
args = parser.parse_args()

# set CUDA devices from input
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices  
chosen_loss = args.chosen_loss

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Configuration -----
headset      = args.train
data_dir     = os.path.join("..", "..", "Data")
sample_len   = 500

num_epochs    = 99
tuning_epochs = 10
models_to_run = ['ResNet1D']

    
if headset == 'Emotive':
    headset_indices = [6, 10, 15, 17, 21, 23, 27, 33, 37, 44, 58, 66, 79, 81]
elif headset == 'DSIVR300':
    headset_indices = [30, 60, 62, 64, 69, 77, 80]
elif headset == 'Muse':
    headset_indices = [4, 12, 46, 56]

# ----- Dataset -----
class EEGDataset(Dataset):
    def __init__(self, path: str, subject_count: int, seed: int = 42):
        self._file   = h5py.File(path, 'r')
        self.data = self._file['data'][:, headset_indices, :] if 'headset_indices' in globals() else self._file['data'][:]
        self.labels  = self._file['labels'][:]

        # Get all unique subjects
        unique_subjects = np.unique(self.labels)

        # Check that requested number is not larger than available subjects
        if subject_count > len(unique_subjects):
            raise ValueError(f"Requested {subject_count} subjects, but only {len(unique_subjects)} available.")

        # Fix the random seed
        rng = random.Random(seed)
        selected_subjects = rng.sample(list(unique_subjects), subject_count)

        # Create subject-to-index mapping only for selected subjects
        self.subj2idx = {s: i for i, s in enumerate(selected_subjects)}

        # Filter the data to include only selected subjects
        mask = np.isin(self.labels, selected_subjects)
        self.data = self.data[mask]
        self.labels = self.labels[mask]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x_raw = self.data[idx]
        y_raw = self.labels[idx]
        x = torch.from_numpy(x_raw).float()
        y = torch.tensor(self.subj2idx[int(y_raw)], dtype=torch.long)
        return x, y

    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()

class EEGDataset_valid(Dataset):
    """
    EEGDataset for validation that returns:
      • x: input tensor (float32)
      • y: subject index tensor (long)
      • s: session ID tensor (long)
    """
    def __init__(self, path: str):
        # Open HDF5 file
        self._file    = h5py.File(path, 'r')
        # Data arrays
        self.data = self._file['data'][:, headset_indices, :] if 'headset_indices' in globals() else self._file['data']
        self.labels   = self._file['labels']     # shape: (N,)
        self.sessions = self._file['sessions']   # shape: (N,)
        # Map raw subject IDs to consecutive indices
        unique_subj    = np.unique(self.labels[:])
        self.subj2idx  = {s: i for i, s in enumerate(unique_subj)}

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # raw data
        x_raw = self.data[idx]       # numpy array
        y_raw = self.labels[idx]     # raw subject ID
        s_raw = self.sessions[idx]   # raw session ID

        # to tensors
        x = torch.from_numpy(x_raw).float()
        y = torch.tensor(self.subj2idx[int(y_raw)], dtype=torch.long)
        s = torch.tensor(int(s_raw), dtype=torch.long)

        return x, y, s

    def __del__(self):
        # ensure file is closed
        if hasattr(self, '_file'):
            self._file.close()
# ----- Models -----
class LSTMModel(nn.Module):
    def __init__(self, channels, hidden, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_size=channels, hidden_size=hidden,
                           num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x.permute(0, 2, 1))
        feat = F.dropout(x[:, -1, :], p=0.3)
        return self.fc(feat)

class GRUModel(nn.Module):
    def __init__(self, channels, hidden, num_classes):
        super().__init__()
        self.rnn = nn.GRU(input_size=channels, hidden_size=hidden,
                          num_layers=2, batch_first=True)
        self.fc  = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x.permute(0, 2, 1))
        feat = F.dropout(x[:, -1, :], p=0.3)
        return self.fc(feat)

# ----- EER -----

def compute_embeddings_dataset(
    ds: Dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128
) -> torch.Tensor:
    """
    Stream input batches from `ds` via DataLoader, keep activations on GPU,
    and only transfer final embeddings to CPU.
    """
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    embs = []

    with torch.no_grad():
        for batch in loader:
            # CHANGED 21: unpack only x; ignore labels/sessions
            x, *rest = batch
            x = x.to(device, non_blocking=True)
            e = model(x)
            embs.append(e.cpu())

    return torch.cat(embs, dim=0)
    
_DISTANCE_FUNCS = {
    "ed": euclidean_distances,
    "cd": cosine_distances,
}

def calculate_similarity_scores_two(
    enroll_emb: np.ndarray,
    y_enroll: np.ndarray,
    verify_emb: np.ndarray,
    y_verify: np.ndarray,
    distance: str = "aa",
    top_k: int = 10
) -> dict:
    """
    For each verification sample and each enrollment class, compute the average of the top_k
    negative distances (i.e. similarity scores), and collect them in a dict keyed by class.
    Returns:
        similarity_results_by_class: { class_label: [ [score, is_target, true_cls, cls, idx], … ] }
    """
    if distance not in _DISTANCE_FUNCS:
        raise ValueError(f"Unsupported distance: {distance}")
    # Compute similarity matrix: higher = more similar
    sim_matrix = -_DISTANCE_FUNCS[distance](verify_emb, enroll_emb)

    unique = np.unique(y_enroll)
    #print(unique)
    class_to_idxs = {cls: np.where(y_enroll == cls)[0] for cls in unique}

    results = defaultdict(list)
    for idx, (scores, true_cls) in enumerate(zip(sim_matrix, y_verify)):
        for cls in unique:
            idxs = class_to_idxs[cls]
            top_scores = np.sort(scores[idxs])[-top_k:]
            avg_score = top_scores.mean()
            label = int(cls == true_cls)
            results[cls].append([avg_score, label, true_cls, cls, idx])

    return results


def compute_eer_metrics(results: np.ndarray) -> Tuple[float, float]:
    genuine = results[results[:, 1] == 1, 0]
    impostor = results[results[:, 1] == 0, 0]
    stats = get_eer_stats(genuine, impostor)
    return stats.eer, stats.fmr100


def get_model(name, emb, chans, samp):
    return {
        'ResNet1D':    lambda: selfeeg.models.ResNet1D(nb_classes=emb, Chans=chans, Samples=samp),
        'ShallowNet':  lambda: selfeeg.models.ShallowNet(nb_classes=emb, Chans=chans, Samples=samp),
        'DeepConvNet': lambda: selfeeg.models.DeepConvNet(nb_classes=emb, Chans=chans, Samples=samp),
        'EEGNet':      lambda: selfeeg.models.EEGNet(nb_classes=emb, Chans=chans, Samples=samp),
        'LSTM':        lambda: LSTMModel(chans, 128, emb),
        'GRU':         lambda: GRUModel(chans, 128, emb)
    }[name]()

def get_loss_and_opts(loss_name, num_classes, emb):
    miner = None
    if loss_name == 'ArcFaceLoss':
        loss_fn = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=emb)
    elif loss_name == 'TripletMarginLoss':
        loss_fn = losses.TripletMarginLoss(margin=0.2)
        miner   = miners.TripletMarginMiner(margin=0.2, type_of_triplets='semihard')
    elif loss_name == 'SupConLoss':
        loss_fn = losses.SupConLoss(temperature=0.1)
        miner   = miners.MultiSimilarityMiner(epsilon=0.1)
    elif loss_name == 'LiftedStructureLoss':
        loss_fn = losses.LiftedStructureLoss()
        miner   = miners.MultiSimilarityMiner(epsilon=0.1)
    elif loss_name == 'SoftTripleLoss':
        loss_fn = losses.SoftTripleLoss(num_classes=num_classes, embedding_size=emb)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
    return loss_fn, miner


def validate_eer(
    model: torch.nn.Module,
    enroll_ds: Dataset,
    verify_ds: Dataset,
    device: torch.device,
    batch_size: int = 128
) -> Tuple[float, float]:
    """
    Compute embeddings on enroll_ds and verify_ds, then return (eer, fmr100).
    """
    model.eval()

    # 1) embeddings
    en_emb = compute_embeddings_dataset(enroll_ds, model, device, batch_size)
    ve_emb = compute_embeddings_dataset(verify_ds, model, device, batch_size)

    # 2) labels
    # CHANGED 23: unpack (x, y, s) tuples
    y_en = np.array([y for _, y, _ in enroll_ds])
    y_ve = np.array([y for _, y, _ in verify_ds])

    # 3) similarity & per-class scores
    sim_dict = calculate_similarity_scores_two(
        en_emb.numpy(), y_en,
        ve_emb.numpy(), y_ve,
        distance="cd", top_k=20
    )
    all_entries = np.vstack(list(sim_dict.values()))

    # 4) final EER + FMR100
    eer, fmr100 = compute_eer_metrics(all_entries)
    return eer, fmr100


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, miner=None):
    model.train()
    total_loss = 0.0
    print("train itrations: ", len(loader))
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            emb_out = model(x)
            loss    = loss_fn(emb_out, y, miner(emb_out, y)) if miner else loss_fn(emb_out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)

def objective(trial, model_name, train_ds, val_ds):
    lr  = trial.suggest_categorical('lr', [5e-5, 1e-4, 1e-3])
    bs  = trial.suggest_categorical('batch_size', [64, 128, 256])
    emb = trial.suggest_categorical('embedding', [64, 128, 256])
    opt_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])  

    loss_fn, miner = get_loss_and_opts(chosen_loss, len(train_ds.subj2idx), emb)
    OptCls         = {'Adam': Adam, 'SGD': SGD, 'RMSprop': RMSprop}[opt_name]

    tr_loader     = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               num_workers=8, pin_memory=True, prefetch_factor=2)

    vl_loader     = DataLoader(val_ds, batch_size=32, shuffle=True,
                               num_workers=8, pin_memory=True, prefetch_factor=2)

    model     = torch.compile(get_model(model_name, emb, train_ds.data.shape[1], sample_len).to(device))
    optimizer = OptCls(list(model.parameters()) +
                       (list(loss_fn.parameters()) if hasattr(loss_fn, 'parameters') else []), lr=lr)
    scaler    = GradScaler()

    print(f"\n*** Training {model_name} ***  {OptCls} {lr} {bs} {emb}")
    for epoch in range(1, tuning_epochs+1):
        train_loss =  train_one_epoch(model, tr_loader, optimizer, scaler, loss_fn, miner)
        val_loss, fmr100 = validate_eer(model, enroll_ds, verify_ds, device)
        trial.report(val_loss, epoch)
        print("epoch: ", epoch,"valiadtion: ", val_loss, "trinloss: ", train_loss)
        if trial.should_prune():
            raise TrialPruned()
    return val_loss


def make_enroll_verify_splits(
    val_ds,
    session_field: str = 'sessions',
    seed: int = 42
) -> Tuple[Subset, Subset]:
    # fetch labels & sessions
    Y_all = val_ds.labels[:]                 
    S_all = val_ds._file[session_field][:]    

    enroll_idxs = []
    verify_idxs = []
    for subj in np.unique(Y_all):
        idxs = np.where(Y_all == subj)[0]
        sess = S_all[idxs]
        # skip subjects without at least two sessions
        if len(np.unique(sess)) < 2:
            continue
        first_sess = sess.min()
        enroll_idxs.extend(idxs[sess == first_sess].tolist())
        verify_idxs.extend(idxs[sess != first_sess].tolist())

    # deterministic shuffle
    rng = np.random.RandomState(seed)
    enroll_idxs = rng.permutation(enroll_idxs)
    verify_idxs = rng.permutation(verify_idxs)

    # build subsets
    enroll_ds = Subset(val_ds, enroll_idxs)
    verify_ds = Subset(val_ds, verify_idxs)
    return enroll_ds, verify_ds

#[8, 16, 32, 64, 128, 230]
if __name__ == '__main__':
    for subject_count in [8, 16, 32, 64, 128]:
        # Load datasets with subject_count argument
        train_ds = EEGDataset(os.path.join(data_dir, "train_raw.h5"), subject_count=subject_count)
        val_ds   = EEGDataset_valid(os.path.join(data_dir, "valid_raw.h5"))

        enroll_ds, verify_ds = make_enroll_verify_splits(val_ds, session_field='sessions')

        all_results = {}
        for name in models_to_run:
            # Use separate database for each subject count
            conn = sqlite3.connect(f"eeg_studies_001.db", timeout=600)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.close()
            storage = RDBStorage(
                url=f"sqlite:///eeg_studies_001.db",
                engine_kwargs={"connect_args": {"timeout": 600, "check_same_thread": False}}
            )

            search_space = {
                'lr':        [5e-5, 1e-4, 1e-3],
                'batch_size':[64, 128, 256],
                'embedding': [64, 128, 256],
                'optimizer': ['Adam','SGD','RMSprop']
            }
            search_iters = int(np.prod([len(v) for v in search_space.values()]))
            print('search_iters: ', search_iters)

            sampler = optuna.samplers.GridSampler(search_space)
            study = optuna.create_study(
                study_name=f"{name}_{chosen_loss}_grid",
                storage=storage,
                load_if_exists=True,
                direction='minimize',
                sampler=sampler
            )
            completed = len(study.trials)
            remaining = search_iters - completed
            print(f"[{name}] {completed} trials completed, {remaining} remaining out of {search_iters} total.")

            if remaining > 0:
                for i in range(remaining):
                    study.optimize(lambda t: objective(t, name, train_ds, val_ds), n_trials=1,
                                   catch=(TrialPruned, optuna.exceptions.StorageInternalError))
                    print(f"[{name}] Trial {completed+i+1}/{search_iters} done — Best ValLoss: {study.best_trial.value:.4f}")
            else:
                print(f"[{name}] All {search_iters} trials already completed; skipping tuning.")

            best     = study.best_trial.params
            best_val = study.best_trial.value
            all_results[name] = best
            print(f"=== Best for {name}: ValLoss={best_val:.4f}, Params={best} ===")

            lr     = best['lr']
            bs     = best['batch_size']
            emb    = best['embedding']
            OptCls = {'Adam': Adam, 'SGD': SGD, 'RMSprop': RMSprop}[best['optimizer']]

            tr_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                   num_workers=8, pin_memory=True, prefetch_factor=2)

            vl_loader = DataLoader(val_ds, batch_size=32, shuffle=True,
                                   num_workers=8, pin_memory=True, prefetch_factor=2)

            model, loss_fn, miner = (
                get_model(name, emb, train_ds.data.shape[1], sample_len).to(device),
                *get_loss_and_opts(chosen_loss, len(train_ds.subj2idx), emb)[:2]
            )
            model = torch.compile(model)
            optimizer = OptCls(list(model.parameters()) +
                               (list(loss_fn.parameters()) if hasattr(loss_fn, 'parameters') else []),
                               lr=lr)
            scaler = GradScaler()

            writer = SummaryWriter(log_dir=os.path.join('runs', f"{name}_{subject_count}"))
            best_val_loss = float('inf')
            log = []

            print(f"\n*** Training {name} for {num_epochs} epochs ***  {OptCls} {lr} {bs} {emb} {miner} {loss_fn} {train_ds.data.shape[1]}")
            os.makedirs('model_3', exist_ok=True)

            for epoch in range(1, num_epochs+1):
                train_loss = train_one_epoch(model, tr_loader, optimizer, scaler, loss_fn, miner)
                val_loss, fmr100 = validate_eer(model, enroll_ds, verify_ds, device)
                print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f} (best={best_val_loss:.4f})")
                log.append((epoch, train_loss, val_loss))
                writer.add_scalars('Loss', {'train': train_loss, 'val': val_loss}, epoch)
                if val_loss < best_val_loss and epoch > 25:
                    best_val_loss = val_loss
                    # Uncomment if you want to save best model:
                    # torch.save(model.state_dict(), os.path.join('model_3', f"{name}_{chosen_loss}_{feature}_{subject_count}_best.pth"))
            writer.close()

            torch.save(model.state_dict(), os.path.join('model_3', f"{name}_{chosen_loss}_{headset}_{subject_count}_final.pth"))

            del model, optimizer, scaler, writer, tr_loader, vl_loader
            torch.cuda.empty_cache()
