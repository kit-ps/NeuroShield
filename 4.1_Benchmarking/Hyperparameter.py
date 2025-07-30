import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
warnings.filterwarnings("ignore")
import os
import random
from collections import defaultdict
from itertools import product
import sqlite3
import h5py
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from pyeer.eer_info import get_eer_stats
from scipy.signal import welch
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances, manhattan_distances
from sklearn.preprocessing import Normalizer, FunctionTransformer, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC, OneClassSVM
from statsmodels.tsa.stattools import yule_walker
import selfeeg
from sklearn.pipeline import Pipeline



# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# -------------------------------
# Model Definitions
# -------------------------------
class LSTMModel(nn.Module):
    """LSTM model for EEG data."""
    def __init__(self, channels, hidden, num_classes):
        super().__init__()
        self.rnn = nn.LSTM(input_size=channels, hidden_size=hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x.permute(0, 2, 1))
        feat = F.dropout(x[:, -1, :], p=0.3)
        return self.fc(feat)

class GRUModel(nn.Module):
    """GRU model for EEG data."""
    def __init__(self, channels, hidden, num_classes):
        super().__init__()
        self.rnn = nn.GRU(input_size=channels, hidden_size=hidden, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        x, _ = self.rnn(x.permute(0, 2, 1))
        feat = F.dropout(x[:, -1, :], p=0.3)
        return self.fc(feat)

def get_model(name: str, emb: int, chans: int, samp: int) -> nn.Module:
    """Retrieve a model instance by name."""
    return {
        'ResNet1D': lambda: selfeeg.models.ResNet1D(nb_classes=emb, Chans=chans, Samples=samp),
        'ShallowNet': lambda: selfeeg.models.ShallowNet(nb_classes=emb, Chans=chans, Samples=samp),
        'DeepConvNet': lambda: selfeeg.models.DeepConvNet(nb_classes=emb, Chans=chans, Samples=samp),
        'EEGNet': lambda: selfeeg.models.EEGNet(nb_classes=emb, Chans=chans, Samples=samp),
        'LSTM': lambda: LSTMModel(chans, 128, emb),
        'GRU': lambda: GRUModel(chans, 128, emb)
    }[name]()

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_and_shuffle_validation_data(file_path: str, seed: int = 42) -> tuple:
    """Load and shuffle validation data from HDF5 file."""
    with h5py.File(file_path, "r") as f:
        X_valid = f['data'][:]
        Y_valid = f['labels'][:]
        S_valid = f['sessions'][:]

    indices = np.arange(X_valid.shape[0])
    np.random.seed(seed)
    np.random.shuffle(indices)
    return X_valid[indices], Y_valid[indices], S_valid[indices]

# -------------------------------
# Embedding Computation
# -------------------------------
def compute_embedding_batch(data: np.ndarray, model: nn.Module, device: torch.device, batch_size: int = 128) -> np.ndarray:
    """Compute embeddings for data in batches."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        tensor_data = torch.tensor(data, dtype=torch.float32, device=device)
        for i in range(0, len(tensor_data), batch_size):
            batch = tensor_data[i:i + batch_size]
            emb = model(batch).cpu().numpy()
            embeddings.append(emb)
    return np.vstack(embeddings) if embeddings else np.empty((0, 0))

def compute_all_embeddings(hyperparams: dict, data: np.ndarray, models: list, loss_functions: list, base_path: str) -> dict:
    """Compute embeddings for all model-loss combinations."""
    all_embeddings = {}
    input_dim, sample_len = 93, 500

    for model_name, loss_fn in product(models, loss_functions):
        study_name = f"{model_name}_{loss_fn}"
        try:
            print(f"\nProcessing {study_name}...")
            loss_base = loss_fn.split("_")[0]
            embedding = int(hyperparams.get(study_name, {}).get("embedding", 256))

            for model_type in ['best', 'final']:
                print(model_type)

                #Re-create model for each 'best'/'final' type
                model = get_model(model_name, embedding, input_dim, sample_len).to(device)
                model_path = f"{base_path}/{model_name}_{loss_base}_{model_type}.pth"

                try:
                    state = torch.load(model_path, weights_only=True)
                    model.load_state_dict(state)
                    model = torch.compile(model)

                except RuntimeError:
                    print(f"[Info] Reloading {model_type} after compiling (state_dict mismatch).")
                    model = get_model(model_name, embedding, input_dim, sample_len).to(device)
                    model = torch.compile(model)
                    state = torch.load(model_path, weights_only=True)
                    model.load_state_dict(state)

                except Exception as e:
                    print(f"[Fatal Error] Failed to load {model_type} model: {e}")
                    continue

                # Compute embeddings
                print(f"Computing embeddings for {study_name} ({model_type})...")
                embeddings = compute_embedding_batch(data, model, device)
                #print("computing: ", embeddings)
                all_embeddings[f"{study_name}_{model_type}"] = {"embeddings": embeddings}

        except Exception as e:
            print(f"[Error] Failed for {study_name}: {str(e)}")

    return all_embeddings

# -------------------------------
# Hyperparameter Loading
# -------------------------------
def load_best_hyperparameters(models: list, loss_functions: list, storage_url: str) -> dict:
    """Load best hyperparameters from Optuna studies."""
    all_best_hyperparameters = {}
    for model, loss_fn in product(models, loss_functions):
        study_name = f"{model}_{loss_fn}"
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            best_trial = study.best_trial
            print(f"\n=== Best Trial Summary for {study_name} ===")
            print(f"Trial Number: {best_trial.number}")
            print(f"Objective Value (Validation Loss): {best_trial.value}")
            print("Hyperparameters:")
            for key, value in best_trial.params.items():
                print(f"  {key}: {value}")
            all_best_hyperparameters[study_name] = best_trial.params
        except KeyError:
            print(f"\n[Warning] Study '{study_name}' not found. Skipping...")
    return all_best_hyperparameters

# -------------------------------
# Evaluation Functions
# -------------------------------
_DISTANCE_FUNCS = {
    "ed": euclidean_distances,
    "cd": cosine_distances,
    "md": manhattan_distances,
}

NormTransformers = {
    "none": FunctionTransformer(lambda X: X),
    "l2": Normalizer(norm="l2"),
    "standard": StandardScaler(),
    "robust": RobustScaler(),
    "minmax": MinMaxScaler(feature_range=(0, 1))
}

def create_validation_folds(unique_subjects: np.ndarray, n_folds: int = 15, n_attackers: int = 7, seed: int = 42) -> list:
    """Create validation folds for evaluation."""
    random.seed(seed)
    folds = []
    for genuine_subject in unique_subjects:
        remaining = [s for s in unique_subjects if s != genuine_subject]
        attackers = random.sample(remaining, n_attackers)
        imposters = [s for s in remaining if s not in attackers]
        folds.append({'genuine': genuine_subject, 'negative': attackers, 'imposters': imposters})
    return folds

def split_enroll_verify_indices(Y: np.ndarray, S: np.ndarray, subject: int, min_session_only: bool = True) -> tuple:
    """Split indices into enrollment and verification sets."""
    subject_idx = np.where(Y == subject)[0]
    subject_sessions = S[subject_idx]
    min_session = np.min(subject_sessions)
    enroll_idx = subject_idx[subject_sessions == min_session] if min_session_only else subject_idx
    verify_idx = subject_idx[subject_sessions != min_session] if min_session_only else []
    return enroll_idx, verify_idx

def calculate_similarity_scores_two(enroll_emb: np.ndarray, y_enroll: np.ndarray, verify_emb: np.ndarray,
                                   y_verify: np.ndarray, distance: str = "ed", top_k: int = 1) -> dict:
    """Compute similarity scores for verification."""
    if distance not in _DISTANCE_FUNCS:
        raise ValueError(f"Unsupported distance: {distance}")
    sim_matrix = -_DISTANCE_FUNCS[distance](verify_emb, enroll_emb)
    unique = np.unique(y_enroll)
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

def compute_eer_metrics(results: np.ndarray) -> tuple[float, float]:
    """Compute EER and FMR100 from similarity scores."""
    genuine = results[results[:, 1] == 1, 0]
    impostor = results[results[:, 1] == 0, 0]
    stats = get_eer_stats(genuine, impostor)
    return stats.eer, stats.fmr100

def fit_and_eer_for_fold(fold: dict, X_emb: np.ndarray, Y: np.ndarray, S: np.ndarray, params: dict) -> float:
    """Compute EER for a single fold."""
    genuine = fold['genuine']
    imposters = fold['imposters']
    enroll_idx, genuine_verify_idx = split_enroll_verify_indices(Y, S, genuine)
    enroll_emb = params['scaler'].fit_transform(X_emb[enroll_idx])
    verify_idx = np.concatenate([genuine_verify_idx, np.where(np.isin(Y, imposters))[0]])
    if len(verify_idx) == 0:
        return 1.0
    verify_emb = params['scaler'].transform(X_emb[verify_idx])
    verify_Y = Y[verify_idx]

    if params['classifier'] in _DISTANCE_FUNCS:
        sim_dict = calculate_similarity_scores_two(enroll_emb, Y[enroll_idx], verify_emb, verify_Y,
                                                  distance=params['classifier'], top_k=params['top_k'])
        arr = np.array(sim_dict[genuine])[:, :2]
        eer, _ = compute_eer_metrics(arr)
        return eer
    else:
        attacker_idx = np.where(np.isin(Y, fold['negative']))[0]
        X_train = np.vstack([enroll_emb, X_emb[attacker_idx]])
        y_train = np.hstack([np.ones(len(enroll_idx)), np.zeros(len(attacker_idx))])
        verify_idx = np.where(np.isin(Y, fold['imposters'] + [genuine]))[0]
        X_verify = X_emb[verify_idx]
        y_verify = (Y[verify_idx] == genuine).astype(int)

        pipeline = Pipeline([
            ('scaler', params['scaler']),
            ('kernel_approximation', params['kernel_approximation']),
            ('classifier', params['classifier'])
        ])

        if isinstance(params['classifier'], OneClassSVM):
            pipeline.fit(enroll_emb)
            preds = pipeline.predict(X_verify)
            probs = (preds == 1).astype(float)
        else:
            pipeline.fit(X_train, y_train)
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(X_verify)[:, 1]
            else:
                dec = pipeline.decision_function(X_verify)
                probs = 1 / (1 + np.exp(-dec))
        stats = get_eer_stats(probs[y_verify == 1], probs[y_verify == 0])
        return stats.eer

def calculate_mean_eer(params: dict, X_emb: np.ndarray, Y: np.ndarray, S: np.ndarray, folds: list) -> float:
    """Calculate mean EER across folds."""
    eers = Parallel(n_jobs=15)(delayed(fit_and_eer_for_fold)(fold, X_emb, Y, S, params) for fold in folds)
    return float(np.mean(eers))

# -------------------------------
# Feature Extraction
# -------------------------------
def compute_psd(signals: np.ndarray, n_fft: int, n_bins: int, n_jobs: int = 20) -> np.ndarray:
    """Compute power spectral density features."""
    def _single_psd(sig):
        _, psd = welch(sig, nperseg=n_fft)
        segments = np.array_split(psd, n_bins)
        return [seg.mean() for seg in segments]
    features = Parallel(n_jobs=n_jobs)(delayed(_single_psd)(sig) for sig in signals)
    return np.array(features)

def compute_ar(signals: np.ndarray, order: int, n_jobs: int = 20) -> np.ndarray:
    """Compute autoregressive features."""
    def _single_ar(sig):
        vec = np.ravel(sig)
        rho, _ = yule_walker(vec, order=order, method='mle')
        return rho
    features = Parallel(n_jobs=n_jobs)(delayed(_single_ar)(sig) for sig in signals)
    return np.array(features)

# -------------------------------
# Optuna Objective
# -------------------------------
def objective(trial: optuna.Trial, Y: np.ndarray, S: np.ndarray, all_embeddings: dict, folds: list) -> float:
    """Optuna objective function for hyperparameter optimization."""
    feature_options = list(all_embeddings.keys()) + ["psd", "ar", "psd+ar"]
    feature_type = trial.suggest_categorical("feature", sorted(feature_options))

    if feature_type in all_embeddings:
        X_emb = all_embeddings[feature_type]["embeddings"]
    elif feature_type == "psd":
        n_fft = trial.suggest_categorical("psd_n_fft", [64, 128, 256, 500])
        n_bins = trial.suggest_categorical("psd_n_bins", [4, 8, 16, 32])
        X_emb = compute_psd(X_valid, n_fft=n_fft, n_bins=n_bins)
    elif feature_type == "ar":
        order = trial.suggest_categorical("ar_order", [1, 2, 4, 8, 16, 32, 64, 128])
        X_emb = compute_ar(X_valid, order=order)
    elif feature_type == "psd+ar":
        n_fft = trial.suggest_categorical("psd_n_fft", [64, 128, 256, 500])
        n_bins = trial.suggest_categorical("psd_n_bins", [4, 8, 16, 32])
        X_psd = compute_psd(X_valid, n_fft=n_fft, n_bins=n_bins)
        order = trial.suggest_categorical("ar_order", [1, 2, 4, 8, 16, 32, 64, 128])
        X_ar = compute_ar(X_valid, order=order)
        X_emb = np.hstack([X_psd, X_ar])

    scaler_name = trial.suggest_categorical("scaler", list(NormTransformers.keys()))
    scaler = NormTransformers[scaler_name]
    classifier_name = trial.suggest_categorical("classifier_name", [
        "random_forest", "sgd_rbf", "logreg", "lda", "svm_linear", "one_class_svm"] + list(_DISTANCE_FUNCS.keys()))

    if classifier_name in _DISTANCE_FUNCS:
        top_k = trial.suggest_categorical("top_k", [1, 5, 10, 20, 30, 40])
        params = {"classifier": classifier_name, "top_k": top_k, "scaler": scaler}
        return calculate_mean_eer(params, X_emb, Y, S, folds)

    kernel_approximation = "passthrough"
    if classifier_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=trial.suggest_categorical("rf_n_estimators", [25, 50, 100, 150]),
            criterion=trial.suggest_categorical("rf_criterion", ["gini", "entropy"]),
            bootstrap=trial.suggest_categorical("rf_bootstrap", [True, False]),
            class_weight="balanced",
            random_state=42,
            n_jobs=10
        )
    elif classifier_name == "sgd_rbf":
        kernel_approximation = RBFSampler(
            gamma=trial.suggest_float("rbf_gamma", 1e-2, 10.0, log=True),
            n_components=trial.suggest_categorical("rbf_n_components", [100, 500, 1000]),
            random_state=42
        )
        classifier = SGDClassifier(
            alpha=trial.suggest_float("sgd_alpha", 1e-5, 1e-2, log=True),
            loss=trial.suggest_categorical("sgd_loss", ["hinge", "log_loss", "modified_huber"]),
            penalty=trial.suggest_categorical("sgd_penalty", ["l2", "l1", "elasticnet"]),
            learning_rate=trial.suggest_categorical("sgd_learning_rate", ["optimal", "invscaling", "adaptive"]),
            eta0=trial.suggest_float("sgd_eta0", 1e-4, 1e-2, log=True),
            power_t=trial.suggest_categorical("sgd_power_t", [0.25, 0.5, 0.75]),
            max_iter=1000,
            tol=1e-3,
            class_weight="balanced",
            random_state=42,
            n_jobs=10
        )
    elif classifier_name == "logreg":
        classifier = LogisticRegression(
            C=trial.suggest_categorical("lr_C", [0.01, 0.1, 1, 5, 10]),
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=10
        )
    elif classifier_name == "svm_linear":
        classifier = LinearSVC(
            C=trial.suggest_categorical("svc_C", [0.01, 0.1, 1, 5, 10]),
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        )
    elif classifier_name == "one_class_svm":
        kernel = trial.suggest_categorical("ocsvm_kernel", ["rbf", "linear", "poly", "sigmoid"])
        nu = trial.suggest_float("ocsvm_nu", 0.01, 0.5, log=True)
        gamma = 'scale' if kernel in ['rbf', 'poly', 'sigmoid'] else 'auto'
        classifier = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
    elif classifier_name == "lda":
        solver = trial.suggest_categorical("lda_solver", ["lsqr", "eigen"])
        shrink = trial.suggest_float("lda_shrinkage", 0.0, 1.0)
        classifier = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrink)

    params = {
        "scaler": scaler,
        "kernel_approximation": kernel_approximation,
        "classifier": classifier
    }
    return calculate_mean_eer(params, X_emb, Y, S, folds)

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    # Load and preprocess data
    X_valid, Y_valid, S_valid = load_and_shuffle_validation_data("../Data/valid_raw.h5")
    unique_subjects = np.unique(Y_valid)
    assert len(unique_subjects) == 15, "Expected 15 subjects in validation set"

    # Define models and loss functions
    models_to_run = ['ResNet1D', 'ShallowNet', 'DeepConvNet', 'EEGNet', 'LSTM', 'GRU']
    loss_functions = ['ArcFaceLoss_grid', 'TripletMarginLoss_grid', 'SupConLoss_grid', 'LiftedStructureLoss_grid', 'SoftTripleLoss_grid']

    # Load hyperparameters
    all_best_hyperparameters = load_best_hyperparameters(models_to_run, loss_functions, "sqlite:///eeg_studies_001.db")

    # Compute embeddings for both 'best' and 'final' models
    all_embeddings = compute_all_embeddings(all_best_hyperparameters, X_valid, models_to_run, loss_functions,
                                           "../4.2_Analysis/Train_Models/model_3")

    # Create validation folds
    folds = create_validation_folds(unique_subjects, n_folds=15, n_attackers=7)

    # Optuna optimization
    conn = sqlite3.connect("Hyperparameter00.db", timeout=600)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()

    storage = optuna.storages.RDBStorage(
        url="sqlite:///Hyperparameter00.db",
        engine_kwargs={"connect_args": {"timeout": 600, "check_same_thread": False}}
    )

    study = optuna.create_study(
        study_name="Hyperparameter00",
        storage=storage,
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=42),
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, Y_valid, S_valid, all_embeddings, folds), n_trials=10000)

    # Print best trial results
    best = study.best_trial
    print(f"Best trial: EER={best.value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")