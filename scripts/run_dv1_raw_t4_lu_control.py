"""Dv1-only, architecture-matched raw-T1..T4 versus soft-LU control.

This is deliberately separate from the historical 66-component control. It
does not alter any retained checkpoint or benchmark file. Run from the project
root with ``python scripts/run_dv1_raw_t4_lu_control.py``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from entanglement_data.ml.invariant_transformer import (  # noqa: E402
    InvariantTokenTransformer, InvariantTransformerConfig,
)
from entanglement_data.ml.raw_siamese import quaternion_to_so3_torch  # noqa: E402
from entanglement_data.observables.feature_spec import FeatureSpec  # noqa: E402
from entanglement_data.observables.local_unitary import feature_slices  # noqa: E402
from lu_control_metrics import benchmark_metrics, load_n16_benchmark  # noqa: E402
from structured_benchmark import load_structured_benchmark  # noqa: E402

SEEDS = (20260829, 20260830, 20260831)
RAW_CHANNELS = 147
MASK_CHANNELS = 5
WIDTH = RAW_CHANNELS + MASK_CHANNELS
N16_H5 = ROOT / "data/n16_production_v1/dataset.h5"
BALANCED_H5 = ROOT / "data/exactXY_balanced_N16_factorized_v1/dataset.h5"
XY32_H5 = ROOT / "data/xy_n32_pbc_matched_exact500_v1/dataset.h5"
DPI32_H5 = ROOT / "data/dpi_n32_general_v1/D_PI_N32.h5"
SEP32_H5 = ROOT / "data/general_separable_ood_n32_v1/dataset.h5"
JOINT_ROOT = ROOT / "models/joint_k16_training_distribution_ablation_v1"
OUT = ROOT / "analysis/dv1_raw_t4_lu_control_v2"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 256
    eval_batch_size: int = 512
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 15
    scheduler_patience: int = 4
    gradient_clip: float = 5.0
    lambda_lu: float = 0.5
    threshold: float = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _expected_masks(n: int, rows: int) -> np.ndarray:
    masks = np.zeros((rows, n, MASK_CHANNELS), np.float32)
    for r in range(1, 5):
        masks[:, : n - r, r - 1] = 1
    masks[:, : n - 2, 4] = 1
    return masks


def full_record_tokens(values: np.ndarray, n: int = 16) -> np.ndarray:
    """Route canonical nonwrapping Pauli records into [T1,T2,T3,T4,masks]."""
    spec = FeatureSpec(n)
    if values.ndim != 2 or values.shape[1] != spec.num_features:
        raise ValueError(f"expected canonical [B,{spec.num_features}]")
    slices = feature_slices(spec)
    batch = len(values)
    out = np.zeros((batch, n, WIDTH), np.float32)
    out[:, :, :3] = values[:, slices.one].reshape(batch, n, 3)
    pair = values[:, slices.two].reshape(batch, -1, 3, 3)
    lookup = np.full((n, n), -1, np.int64)
    lookup[slices.pair_left, slices.pair_right] = np.arange(len(slices.pair_left))
    offset = 3
    for r in range(1, 5):
        indices = lookup[np.arange(n-r), np.arange(r,n)]
        if (indices < 0).any():
            raise RuntimeError(f"missing nonwrapping T2 r={r}")
        out[:, :n-r, offset:offset+9] = pair[:, indices].reshape(batch,n-r,9)
        offset += 9
    out[:, :n-2, 39:66] = values[:, slices.three].reshape(batch,n-2,27)
    out[:, :n-3, 66:147] = values[:, slices.four].reshape(batch,n-3,81)
    out[:, :, RAW_CHANNELS:] = _expected_masks(n,batch)
    if not np.isfinite(out).all():
        raise FloatingPointError("nonfinite canonical Pauli correlator")
    return out


def structured_tokens(path: Path, prefix: str) -> np.ndarray:
    with h5py.File(path, "r") as h5:
        for name in ("T1", "T2", "T3", "T4"):
            if f"{prefix}/{name}" not in h5:
                raise KeyError(f"{path}: missing real raw {name}; zero filling forbidden")
        one = h5[f"{prefix}/T1"][:]
        pair = h5[f"{prefix}/T2"][:]
        triple = h5[f"{prefix}/T3"][:]
        four = h5[f"{prefix}/T4"][:]
        batch,n,_ = one.shape
        if pair.shape != (batch,n,4,3,3) or triple.shape != (batch,n,3,3,3) or four.shape != (batch,n,3,3,3,3):
            raise ValueError(f"{path}: raw tensor shapes differ from schema")
        expected = _expected_masks(n,batch)
        for key, valid in (("T2_mask",expected[:,:,:4]),("T3_mask",expected[:,:,4]),("T4_mask",expected[:,:,2])):
            if f"{prefix}/{key}" in h5:
                actual = np.asarray(h5[f"{prefix}/{key}"][:])
                if actual.shape != valid.shape or not np.array_equal(actual > 0, valid > 0):
                    raise RuntimeError(f"{path}: {key} differs from the nonwrapping support schema")
        out = np.zeros((batch,n,WIDTH),np.float32)
        out[:,:,:3] = one
        for r in range(1,5):
            start = 3+9*(r-1)
            out[:,:n-r,start:start+9] = pair[:,:n-r,r-1].reshape(batch,n-r,9)
        out[:,:n-2,39:66] = triple[:,:n-2].reshape(batch,n-2,27)
        out[:,:n-3,66:147] = four[:,:n-3].reshape(batch,n-3,81)
        out[:,:,RAW_CHANNELS:] = expected
    if not np.isfinite(out).all():
        raise FloatingPointError(f"{path}: nonfinite raw correlator")
    return out


def rotate_tokens(tokens: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """One independent SO(3) per site; masks pass through unchanged."""
    batch,n,width = tokens.shape
    if width != WIDTH or rotations.shape != (batch,n,3,3):
        raise ValueError("raw token or local rotation shape mismatch")
    out = torch.zeros_like(tokens)
    out[:,:,:3] = torch.einsum("bqai,bqi->bqa", rotations,tokens[:,:,:3])
    for r in range(1,5):
        start = 3+9*(r-1)
        block = tokens[:,:n-r,start:start+9].reshape(batch,n-r,3,3)
        out[:,:n-r,start:start+9] = torch.einsum(
            "bqai,bqcj,bqij->bqac",rotations[:,:n-r],rotations[:,r:],block
        ).reshape(batch,n-r,9)
    triple = tokens[:,:n-2,39:66].reshape(batch,n-2,3,3,3)
    out[:,:n-2,39:66] = torch.einsum(
        "nqai,nqbj,nqck,nqijk->nqabc",
        rotations[:,:n-2],rotations[:,1:n-1],rotations[:,2:],triple,
    ).reshape(batch,n-2,27)
    four = tokens[:,:n-3,66:147].reshape(batch,n-3,3,3,3,3)
    out[:,:n-3,66:147] = torch.einsum(
        "nqai,nqbj,nqck,nqdl,nqijkl->nqabcd",
        rotations[:,:n-3],rotations[:,1:n-2],rotations[:,2:n-1],rotations[:,3:],four,
    ).reshape(batch,n-3,81)
    out[:,:,RAW_CHANNELS:] = tokens[:,:,RAW_CHANNELS:]
    return out


def normalization(train: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Match Joint-K16: valid training anchors only; masks left at 0/1."""
    if train.ndim != 3 or train.shape[-1] != WIDTH:
        raise ValueError("unexpected training token shape")
    count = np.zeros(WIDTH,np.int64)
    total = np.zeros(WIDTH,np.float64)
    squared = np.zeros(WIDTH,np.float64)
    n = train.shape[1]
    for left,right,span in ((0,3,n),(3,12,n-1),(12,21,n-2),(21,30,n-3),(30,39,n-4),(39,66,n-2),(66,147,n-3)):
        flat = np.asarray(train[:,:span,left:right],np.float64).reshape(-1,right-left)
        count[left:right] = len(flat)
        total[left:right] = flat.sum(axis=0)
        squared[left:right] = np.square(flat).sum(axis=0)
    mean = np.zeros(WIDTH,np.float32)
    std = np.ones(WIDTH,np.float32)
    mean[:RAW_CHANNELS] = (total[:RAW_CHANNELS]/count[:RAW_CHANNELS]).astype(np.float32)
    variance = np.maximum(squared[:RAW_CHANNELS]/count[:RAW_CHANNELS]-np.square(mean[:RAW_CHANNELS]),1e-12)
    std[:RAW_CHANNELS] = np.sqrt(variance).astype(np.float32)
    return mean,std


def make_cache() -> tuple[np.ndarray,dict[str,np.ndarray],dict[str,np.ndarray]]:
    cache = OUT / "cache"
    cache.mkdir(parents=True,exist_ok=True)
    path = cache / "Dv1_raw_T1T2T3T4.npy"
    with h5py.File(N16_H5,"r") as h5:
        source = h5["features/X"]
        if source.shape != (50000,FeatureSpec(16).num_features):
            raise RuntimeError("Dv1 source schema changed")
        labels = h5["labels/y"][:].astype(np.float32)
        split = h5["metadata/split"].asstr()[:]
        parent = h5["metadata/parent_state_id"].asstr()[:]
        if not path.exists():
            cache_data = np.lib.format.open_memmap(path,"w+",dtype=np.float32,shape=(len(labels),16,WIDTH))
            for start in range(0,len(labels),256):
                stop = min(start+256,len(labels))
                cache_data[start:stop] = full_record_tokens(source[start:stop])
            cache_data.flush()
    data = np.load(path,mmap_mode="r")
    if data.shape != (50000,16,WIDTH):
        raise RuntimeError("stale raw-T4 cache")
    rows = {name:np.flatnonzero(split==name) for name in ("train","validation","test")}
    if [len(rows[name]) for name in rows] != [35000,7500,7500]:
        raise RuntimeError("Dv1 split counts changed")
    parent_sets = [set(parent[idx].tolist()) for idx in rows.values()]
    if any(parent_sets[i]&parent_sets[j] for i,j in ((0,1),(0,2),(1,2))):
        raise RuntimeError("parent leakage between splits")
    for name,index in rows.items():
        if labels[index].sum()!=len(index)/2:
            raise RuntimeError(f"{name} class balance changed")
    return data,rows,{"labels":labels,"source_hash":sha256(N16_H5),"cache_hash":sha256(path)}


@torch.no_grad()
def predict(model: nn.Module, tokens: np.ndarray, mean: np.ndarray, std: np.ndarray,
            device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    scores=[]
    mu = torch.as_tensor(mean,device=device).view(1,1,-1)
    sd = torch.as_tensor(std,device=device).view(1,1,-1)
    for start in range(0,len(tokens),batch_size):
        x=torch.as_tensor(np.asarray(tokens[start:start+batch_size]),device=device)
        scores.append(torch.sigmoid(model((x-mu)/sd)).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def train_one(mode: str, seed: int, data: np.ndarray, rows: dict[str,np.ndarray],
              labels: np.ndarray, mean: np.ndarray, std: np.ndarray,
              device: torch.device, cfg: TrainConfig) -> tuple[nn.Module,dict]:
    path = OUT / f"{mode}_seed{seed}"
    path.mkdir(parents=True,exist_ok=True)
    checkpoint_path=path/"best.pt"
    seed_all(seed)
    model=InvariantTokenTransformer(InvariantTransformerConfig(WIDTH)).to(device)
    parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameters!=500737:
        raise RuntimeError(f"matched raw model has {parameters} parameters")
    if checkpoint_path.exists():
        checkpoint=torch.load(checkpoint_path,map_location=device)
        if checkpoint["mode"]!=mode or checkpoint["seed"]!=seed:
            raise RuntimeError("checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model_state_dict"])
        return model,json.loads((path/"training_report.json").read_text())
    x_train = np.asarray(data[rows["train"]],np.float32)
    y_train = labels[rows["train"]]
    x_valid = data[rows["validation"]]
    y_valid = labels[rows["validation"]]
    loader=DataLoader(TensorDataset(torch.from_numpy(x_train),torch.from_numpy(y_train)),
        batch_size=cfg.batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),
        num_workers=0,pin_memory=device.type=="cuda")
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.5,patience=cfg.scheduler_patience)
    scaler=torch.cuda.amp.GradScaler(enabled=device.type=="cuda")
    bce=nn.BCEWithLogitsLoss()
    mu=torch.as_tensor(mean,device=device).view(1,1,-1)
    sd=torch.as_tensor(std,device=device).view(1,1,-1)
    history=[]
    best=math.inf
    stale=0
    started=time.perf_counter()
    for epoch in range(1,cfg.max_epochs+1):
        model.train()
        train_loss=0.0
        for raw,y in loader:
            raw=raw.to(device,non_blocking=True)
            y=y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
                z=model((raw-mu)/sd)
                if mode=="soft_lu":
                    quaternion=torch.randn((len(raw),raw.shape[1],4),device=device)
                    rotations=quaternion_to_so3_torch(quaternion)
                    rotated=rotate_tokens(raw,rotations)
                    zr=model((rotated-mu)/sd)
                    supervised=0.5*(bce(z,y)+bce(zr,y))
                    consistency=torch.mean(torch.abs(torch.sigmoid(z)-torch.sigmoid(zr)))
                    loss=supervised+cfg.lambda_lu*consistency
                else:
                    loss=bce(z,y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),cfg.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            train_loss+=float(loss.detach())*len(y)
        probability=predict(model,x_valid,mean,std,device,cfg.eval_batch_size)
        eps=1e-12
        valid_bce=float(-np.mean(y_valid*np.log(np.clip(probability,eps,1-eps))+(1-y_valid)*np.log(np.clip(1-probability,eps,1-eps))))
        scheduler.step(valid_bce)
        row={"epoch":epoch,"training_total_loss":train_loss/len(x_train),"validation_BCE":valid_bce,
             "validation_AUROC":float(roc_auc_score(y_valid,probability)),"lr":optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(json.dumps({"mode":mode,"seed":seed,**row,"stale":stale}),flush=True)
        if valid_bce<best-1e-7:
            best=valid_bce
            stale=0
            torch.save({"model_state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},
                "mode":mode,"seed":seed,"epoch":epoch,"validation_BCE":best,
                "input_width":WIDTH,"parameter_count":parameters,"config":asdict(cfg),
                "normalization_mean":mean,"normalization_std":std},checkpoint_path)
        else:
            stale+=1
        if stale>=cfg.patience:
            break
    write_csv(path/"training_history.csv",history)
    checkpoint=torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    report={"mode":mode,"seed":seed,"parameter_count":parameters,"training_sources":["D_v1"],
        "train_rows":len(x_train),"validation_rows":len(x_valid),"best_epoch":checkpoint["epoch"],
        "best_validation_BCE":best,"checkpoint_sha256":sha256(checkpoint_path),
        "wall_seconds":time.perf_counter()-started,"training_config":asdict(cfg),
        "normalization":"Dv1 training split valid anchors only; mask mean=0 std=1"}
    write_json(path/"training_report.json",report)
    return model,report


def benchmark_inputs() -> dict[str,tuple[np.ndarray,dict]]:
    with h5py.File(BALANCED_H5,"r") as h5:
        balanced=full_record_tokens(h5["features/X"][:])
    xy=load_structured_benchmark(XY32_H5,"xy")
    dpi=load_structured_benchmark(DPI32_H5,"dpi")
    sep=structured_tokens(SEP32_H5,"raw")
    n=len(sep)
    sep_labels={"sample_id":np.arange(n),"separable":np.ones(n,bool),"entangled":np.zeros(n,bool),
        "E2":np.zeros(n,bool),"E3":np.zeros(n,bool),"E4":np.zeros(n,bool),"unknown":np.zeros(n,bool)}
    with h5py.File(SEP32_H5,"r") as h5:
        if "meta/sample_id" in h5:
            sep_labels["sample_id"]=h5["meta/sample_id"][:]
    return {
        "N16_balanced":(balanced,load_n16_benchmark(BALANCED_H5)),
        "N32_XY":(structured_tokens(XY32_H5,"features"),xy[3]),
        "N32_DPI":(structured_tokens(DPI32_H5,"raw"),dpi[3]),
        "N32_GeneralSep":(sep,sep_labels),
    }


def metrics_for(probability:np.ndarray, labels:dict) -> dict:
    if np.asarray(labels["entangled"],bool).any():
        result=benchmark_metrics(probability,labels,0.5)
        return {k:result[k] for k in ("AUROC","E2_recall","E_gt_2_recall","separable_FPR","separable_count","E2_count","E_gt_2_count","unknown_count")}
    sep=np.asarray(labels["separable"],bool)
    return {"AUROC":None,"E2_recall":None,"E_gt_2_recall":None,
        "separable_FPR":float(np.mean(probability[sep]>=0.5)),"separable_count":int(sep.sum()),
        "E2_count":0,"E_gt_2_count":0,"unknown_count":0}


def old_joint_scores(benchmark:str, labels:dict) -> list[np.ndarray]:
    names={"N16_balanced":"scores_N16_XY_balanced.csv","N32_XY":"scores_N32_XY_matched.csv",
        "N32_DPI":"scores_N32_DPI_general.csv","N32_GeneralSep":"scores_N32_general_separable.csv"}
    expected=np.asarray(labels["sample_id"],np.int64)
    result=[]
    for seed in SEEDS:
        directory=JOINT_ROOT/f"RoPETransformer_JointK16_Dv1_only_seed{seed}"
        report=json.loads((directory/"report.json").read_text(encoding="utf-8"))
        if report["training_sources"]!=["D_v1"] or report["parameter_count"]!=483969:
            raise RuntimeError(f"Joint-K16 seed {seed} is not canonical Dv1-only")
        with (directory/names[benchmark]).open(newline="",encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        if len(rows)!=len(expected) or not np.array_equal(
            np.asarray([int(r["row"]) for r in rows]),np.arange(len(expected))):
            raise RuntimeError(f"Joint-K16 {benchmark} row order or count differs")
        if "sample_id" in rows[0]:
            ids=np.asarray([int(r["sample_id"]) for r in rows])
            if not np.array_equal(ids,expected):
                raise RuntimeError(f"Joint-K16 {benchmark} sample IDs differ")
        elif benchmark!="N32_GeneralSep":
            raise RuntimeError(f"Joint-K16 {benchmark} lacks sample IDs")
        result.append(np.asarray([float(r["probability"]) for r in rows]))
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--modes",nargs="+",choices=("raw","soft_lu"),default=["raw","soft_lu"])
    parser.add_argument("--seeds",nargs="+",type=int,default=list(SEEDS))
    args=parser.parse_args()
    if not set(args.seeds)<=set(SEEDS):
        raise ValueError("only predeclared seeds are allowed")
    OUT.mkdir(parents=True,exist_ok=True)
    data,rows,source=make_cache()
    mean,std=normalization(np.asarray(data[rows["train"]]))
    np.save(OUT/"train_mean.npy",mean)
    np.save(OUT/"train_std.npy",std)
    manifest={"physical_channels":RAW_CHANNELS,"mask_channels":MASK_CHANNELS,"input_width":WIDTH,
        "order":"T1[3],T2[r1..r4,9 each],T3[27],T4[81],geometry masks[r1..r4,T3]",
        "four_body_support":"nonwrapping contiguous i..i+3; shares r=3 validity mask",
        "split_counts":{k:len(v) for k,v in rows.items()},"Dv1_source_sha256":source["source_hash"],
        "Dv1_raw_cache_sha256":source["cache_hash"],"evaluation_used_for_normalization":False,
        "historical_results_overwritten":False,"seeds":list(SEEDS)}
    write_json(OUT/"experiment_manifest.json",manifest)
    if args.prepare_only:
        print(json.dumps(manifest),flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the predeclared experiment")
    torch.set_num_threads(2)
    device=torch.device("cuda")
    cfg=TrainConfig()
    benchmarks=benchmark_inputs()
    per_seed=[]
    ensembles=[]
    for mode in (*args.modes,"joint_k16"):
        for benchmark,(tokens,labels) in benchmarks.items():
            probabilities=[]
            for seed in args.seeds:
                if mode=="joint_k16":
                    continue
                model,report=train_one(mode,seed,data,rows,source["labels"],mean,std,device,cfg)
                probability=predict(model,tokens,mean,std,device,cfg.eval_batch_size)
                probabilities.append(probability)
                per_seed.append({"model":mode,"seed":seed,"benchmark":benchmark,**metrics_for(probability,labels)})
            if mode=="joint_k16":
                if tuple(args.seeds)!=SEEDS:
                    continue
                probabilities=old_joint_scores(benchmark,labels)
                for seed,probability in zip(SEEDS,probabilities):
                    per_seed.append({"model":mode,"seed":seed,"benchmark":benchmark,**metrics_for(probability,labels)})
            if probabilities:
                average=np.stack(probabilities).mean(axis=0)
                ensembles.append({"model":mode,"benchmark":benchmark,"seeds":len(probabilities),**metrics_for(average,labels)})
                write_csv(OUT/"scores"/f"{mode}_{benchmark}.csv",[
                    {"sample_id":int(labels["sample_id"][i]),"ensemble_probability":float(average[i]),
                     **{f"probability_seed{seed}":float(p[i]) for seed,p in zip(args.seeds,probabilities)}}
                    for i in range(len(average))])
            write_csv(OUT/"per_seed_metrics.csv",per_seed)
            write_csv(OUT/"ensemble_metrics.csv",ensembles)
    aggregate=[]
    for mode in (*args.modes,"joint_k16"):
        for benchmark in benchmarks:
            selected=[row for row in per_seed if row["model"]==mode and row["benchmark"]==benchmark]
            if not selected:
                continue
            result={"model":mode,"benchmark":benchmark,"seeds":len(selected)}
            for metric in ("AUROC","E2_recall","E_gt_2_recall","separable_FPR"):
                values=np.asarray([row[metric] for row in selected if row[metric] is not None],np.float64)
                result[f"{metric}_mean"]=float(values.mean()) if len(values) else None
                result[f"{metric}_sample_std"]=float(values.std(ddof=1)) if len(values)>1 else None
            aggregate.append(result)
    write_csv(OUT/"per_seed_summary.csv",aggregate)
    print(json.dumps({"status":"complete","per_seed_rows":len(per_seed),"ensemble_rows":len(ensembles)}),flush=True)


if __name__=="__main__":
    main()
