# Imports
from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer, AutoModelForSequenceClassification,EarlyStoppingCallback,default_data_collator
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Bio import SeqIO
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
from transformers import set_seed as hf_set_seed
import pandas as pd
import os
import gc
from dataclasses import dataclass

SEEDS = [42, 123, 999]
# SEEDS = [42]
def set_seed(seed=42):

    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDA deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = False
    # torch.backends.cudnn.benchmark = True

    # HuggingFace
    hf_set_seed(seed)

    # Python hash
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Seed set to {seed}")

MAX_LEN = 256
n_train = 400

RUNS_DIR = "/results"

MODEL_DIR = "/NT_v2_500m_multi_species"
MODEL_NAME = "NT_v2_500m_multi_species"

from modeling_esm import EsmForSequenceClassification, EsmClassificationHead
from esm_config import EsmConfig

checkpoint_dir = MODEL_DIR

def load_model_and_tokenizer_from_checkpoint():
    config = EsmConfig.from_pretrained(checkpoint_dir)
    model = EsmForSequenceClassification.from_pretrained(checkpoint_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

from datasets import load_dataset, Dataset
from Bio import SeqIO

species_list = ["Human"]
fasta_paths = {
    "Human": {"train": "/datasets/multi_promoters/H_sapiens_Human_promoters/H_sapiens_Human_promoters_train.fna", "test": "/datasets/multi_promoters/H_sapiens_Human_promoters/H_sapiens_Human_promoters_test.fna"},
}

def load_fasta_with_label(filepath):
    seqs, labs = [], []
    for record in SeqIO.parse(filepath, "fasta"):
        seqs.append(str(record.seq))
        labs.append(int(record.description.split("|")[-1]))
    return seqs, labs

def build_datasets_dict(tokenizer, seed):

    datasets_dict = {}

    def tokenize_function(examples):
        return tokenizer(
            examples["data"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    for sp in species_list:
        if sp not in fasta_paths:
            raise ValueError(f"fasta_paths does not {sp}")
        if "train" not in fasta_paths[sp] or "test" not in fasta_paths[sp]:
            raise ValueError(f"fasta_paths[{sp}] must contain train/test")

        print(f"Processing species: {sp}")

        train_seq, train_lab = load_fasta_with_label(fasta_paths[sp]["train"])
        test_seq,  test_lab  = load_fasta_with_label(fasta_paths[sp]["test"])

        train_seq, val_seq, train_lab, val_lab = train_test_split(
            train_seq,
            train_lab,
            test_size=0.05,
            random_state=seed,
            stratify=train_lab,
        )

        datasets_dict[sp] = {
            "train_raw": train_seq,
            "val_raw": val_seq,
            "test_raw": test_seq,
        }

        ds_train = Dataset.from_dict({"data": train_seq, "labels": train_lab})
        ds_val   = Dataset.from_dict({"data": val_seq,   "labels": val_lab})
        ds_test  = Dataset.from_dict({"data": test_seq,  "labels": test_lab})

        ds_train = ds_train.map(tokenize_function, batched=True, remove_columns=["data"])
        ds_val   = ds_val.map(tokenize_function, batched=True, remove_columns=["data"])
        ds_test  = ds_test.map(tokenize_function, batched=True, remove_columns=["data"])

        datasets_dict[sp].update({
                "train": ds_train,
                "val": ds_val,
                "test": ds_test
            })

    return datasets_dict

def subset_hf_dataset_stratified(ds_train, n, seed=42, label_col="labels"):

    if n is None or n >= len(ds_train):
        return ds_train

    rng = np.random.RandomState(seed)
    labels = np.array(ds_train[label_col])

    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]

    half = n // 2
    extra = n - 2 * half  # 0 or 1

    rng.shuffle(pos)
    rng.shuffle(neg)

    take_pos = half + extra
    take_neg = half

    if len(pos) < take_pos or len(neg) < take_neg:
        raise ValueError(f"Sample size is insufficient: pos={len(pos)}, neg={len(neg)}, need pos={take_pos}, neg={take_neg}")

    idxs = np.concatenate([pos[:take_pos], neg[:take_neg]])
    rng.shuffle(idxs)
    return ds_train.select(idxs.tolist()), idxs

def compute_metrics(eval_pred):
    preds, labels = eval_pred

    if isinstance(preds, (tuple, list)):
        preds = preds[0]

    y_pred = np.argmax(preds, axis=-1)

    acc = accuracy_score(labels, y_pred)
    f1  = f1_score(labels, y_pred, average="macro")
    p   = precision_score(labels, y_pred, average="macro", zero_division=0)
    r   = recall_score(labels, y_pred, average="macro", zero_division=0)

    return {"f1": f1, "accuracy": acc, "precision": p, "recall": r}


def make_training_args_no_ckpt(num_epochs, metric_for_best="f1"):
    return TrainingArguments(
        learning_rate=LR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BS,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=False,
        greater_is_better=True,
        metric_for_best_model="eval_f1",
    )

def freeze_backbone_only_train_head(model, reset_head=True):
    for p in model.esm.parameters():
        p.requires_grad = False
    if reset_head:
        model.classifier = EsmClassificationHead(model.config).to(next(model.parameters()).device)
    for p in model.classifier.parameters():
        p.requires_grad = True

# ==========================================================
# 10) Adapter 
# ==========================================================
class BottleneckAdapter(nn.Module):
    def __init__(self, d_model, bottleneck=BOTTLENECK, dropout=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, d_model, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, h):
        return self.up(self.drop(self.act(self.down(h))))  # Δh

class GumbelSigmoid(nn.Module):
    def __init__(self, tau=0.7):
        super().__init__()
        self.tau = tau

    def forward(self, logits):
        # Gumbel noise
        noise = torch.rand_like(logits)
        noise = torch.log(noise + NOISE) - torch.log(1 - noise + NOISE)

        return torch.sigmoid((logits + noise) / self.tau)

class SparseLayerGate(nn.Module):
    def __init__(self, d_model, num_layers):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, num_layers)
        )

        self.gumbel = GumbelSigmoid(tau=0.7)

    def forward(self, hidden_states):
        # CLS token
        cls = hidden_states[:, 0]  # [B, D]

        logits = self.net(cls)     # [B, L]
        gates = self.gumbel(logits)

        return gates

class ConcatFusion(nn.Module):
    def __init__(self, d_model, bottleneck=BOTTLENECK, dropout=0.1, use_residual=True):
        super().__init__()
        self.adapter = BottleneckAdapter(d_model, bottleneck=bottleneck, dropout=dropout)
        self.proj = nn.Linear(2 * d_model, d_model, bias=False)
        nn.init.zeros_(self.proj.weight)
        self.use_residual = use_residual

    def forward(self, h):
        delta = self.adapter(h)                    # [B,L,D]
        cat = torch.cat([h, delta], dim=-1)        # [B,L,2D]
        fused = self.proj(cat)                     # [B,L,D]
        return (h + fused) if self.use_residual else fused

class GatedConcatFusion(nn.Module):
    def __init__(self, d_model, bottleneck=BOTTLENECK, dropout=0.1):
        super().__init__()

        self.fusion = ConcatFusion(d_model, bottleneck, dropout)

    def forward(self, h, gate):
        """
        h:    [B,L,D]
        gate: [B] or [B,1,1]
        """
        fused = self.fusion(h)

        # reshape gate
        if gate.dim() == 1:
            gate = gate.view(-1, 1, 1)

        return h + gate * fused

def attach_sparse_layer_selection(model, K=None, bottleneck=BOTTLENECK, dropout=0.1):

    device = next(model.parameters()).device
    d_model = model.config.hidden_size

    layers = model.esm.encoder.layer
    num_layers = len(layers)

    # ===== 1. gate net =====
    model._gate_net = SparseLayerGate(d_model, num_layers).to(device)

    # ===== 2. fusion modules =====
    model._fusions = nn.ModuleList([
        GatedConcatFusion(d_model, bottleneck, dropout)
        for _ in range(num_layers)
    ]).to(device)

    handles = []

    def make_hook(layer_idx):

        fusion = model._fusions[layer_idx]

        def hook(module, inputs, output):

            if isinstance(output, tuple):
                h = output[0]
                rest = output[1:]
            else:
                h = output
                rest = None

            if layer_idx == 0:
                gates = model._gate_net(h)  # [B,L]
                model._current_gates = gates

            gates = model._current_gates

            # ===== optional layer constraint =====
            if K is not None and layer_idx >= K:
                return output

            # ===== apply gate =====
            g_l = gates[:, layer_idx].view(-1, 1, 1)

            h_new = fusion(h, g_l)

            if rest is not None:
                return (h_new,) + rest
            return h_new

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(i)))

    for p in model._gate_net.parameters():
        p.requires_grad = True

    for p in model._fusions.parameters():
        p.requires_grad = True

    return handles

def cleanup(trainer=None, model=None):
    try:
        if trainer is not None:
            trainer.model = None
            trainer.optimizer = None
            trainer.lr_scheduler = None
            trainer.train_dataloader = None
            trainer.eval_dataloader = None
    except Exception:
        pass

    if trainer is not None:
        del trainer
    if model is not None:
        del model

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def gate_regularization_loss(model):

    gates = model._current_gates  # [B, L]

    # 1. L1 sparsity
    l1 = gates.mean()

    # 2. encourage few active layers
    target_k = 3
    l2 = torch.abs(gates.sum(dim=1) - target_k).mean()

    # 3. entropy
    entropy = -(gates * torch.log(gates + EPS) +
                (1 - gates) * torch.log(1 - gates + EPS)).mean()

    return 0.05 * l1 + 0.1 * l2 + 0.01 * entropy

class MultiSpeciesGatedTrainer(Trainer):

    def __init__(self, *args, save_gate_dir=None, visualize_gate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_gate_dir = save_gate_dir
        if save_gate_dir is not None:
            os.makedirs(save_gate_dir, exist_ok=True)

        self.visualize_gate = visualize_gate

        self._all_gates = []


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        outputs = model(**inputs)
        loss = outputs.loss

        if hasattr(model, "_current_gates"):
            reg = gate_regularization_loss(model)
            loss = loss + reg

            self._all_gates.append(model._current_gates.detach().cpu())

        return (loss, outputs) if return_outputs else loss


def run_all_steps_for_species_and_n(datasets_dict, tokenizer, sp, n_train, seed, output_root):

    ds_train_all = datasets_dict[sp]["train"]
    ds_val = datasets_dict[sp]["val"]
    ds_test = datasets_dict[sp]["test"]

    train_seq = datasets_dict[sp]["train_raw"]

    ds_train, train_indices = subset_hf_dataset_stratified(ds_train_all, n=n_train, seed=seed, label_col="labels")
    train_seq = [train_seq[i] for i in train_indices]

    print("\n" + "=" * 110)
    print(f"species：{sp} | train_len={len(ds_train)}")
    print("=" * 110)

    model, _, _ = load_model_and_tokenizer_from_checkpoint()
    freeze_backbone_only_train_head(model, reset_head=True)
    handles = attach_sparse_layer_selection(model, bottleneck=BOTTLENECK, dropout=ADAPTER_DROPOUT)

    args = make_training_args_no_ckpt(EPOCHS_ADAPTER)
    trainer = MultiSpeciesGatedTrainer(
        model=model, args=args,
        train_dataset=ds_train, eval_dataset=ds_val,
        tokenizer=tokenizer, data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)],
    )
    trainer.train()

    res = trainer.evaluate(eval_dataset=ds_test)
    
    print("[GATE all]", res)

    for h in handles:
        h.remove()
    cleanup(trainer, model)

if __name__ == "__main__":

    _, tokenizer, _ = load_model_and_tokenizer_from_checkpoint()
    for seed in SEEDS:
        set_seed(seed)
        datasets_dict = build_datasets_dict(tokenizer, seed)
        seed_output_root = os.path.join(RUNS_DIR,f"seed_{seed}")
        print("\n" + "#" * 110)
        print(f"########### seed={seed} ###########")
        print("#" * 110)
        for sp in species_list:
            run_all_steps_for_species_and_n(datasets_dict, tokenizer, sp, n_train, seed, seed_output_root)

