# -*- coding: utf-8 -*-
"""PESGM 2026 analysis pipeline utilities."""


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import h5py
from scipy.io import loadmat

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# ============== Config (train + test) ==============
TRAIN_FILE_PATHS = [
    "gfmi1_impedance_dataset.mat",
    "gfmi2_impedance_dataset.mat",
    "gfmi3_impedance_dataset.mat",
    "gfli1_impedance_dataset.mat",
    "gfli2_impedance_dataset.mat",
    "gfli3_impedance_dataset.mat",
]

TEST_FILE_PATHS = [
    "gfmi1_test_impedance_dataset.mat",
    "gfmi2_test_impedance_dataset.mat",
    "gfmi3_test_impedance_dataset.mat",
    "gfli1_test_impedance_dataset.mat",
    "gfli2_test_impedance_dataset.mat",
    "gfli3_test_impedance_dataset.mat",
]

NBINS = 100
# Output
OUT_IMG = f"bode_Ydd_kmeans_lines.png"
OUT_SUMMARY = f"cluster_summary_Ydd.csv"
LABEL_KEY = 'Y_Y'

# ============== IO helpers ==============

def _fix_shape(arr, expected_cols=None):
    """
    Ensure 2D shape. If expected_cols is given, transpose when needed.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if expected_cols is not None:
        if arr.shape[1] != expected_cols and arr.shape[0] == expected_cols:
            arr = arr.T
    return arr

def _extract_from_h5(f, label_key="Y_Y"):
    """
    Try to extract X (4 cols) and Y (8 cols) from an h5py.File handle.
    Expected structure:
      /Dataset/X, /Dataset/Y_Y (or label_key)
    Fallbacks:
      /X, /Y_Y at root.
    """
    # normalize key candidates
    def get_node(g, key):
        return g[key] if key in g else None

    # 1) common path: group 'Dataset'
    g = get_node(f, "Dataset")
    if g is None:
        # try lowercase
        g = get_node(f, "dataset")

    if g is not None:
        X = _fix_shape(g["X"][()], expected_cols=4)
        Ysrc = g[label_key] if label_key in g else g.get("Y_Y")
        if Ysrc is None:
            raise KeyError("Neither label_key nor 'Y_Y' found under /Dataset")
        Y = _fix_shape(Ysrc[()], expected_cols=8)
        return X, Y

    # 2) fallback: variables at file root
    Xnode = get_node(f, "X")
    Ynode = get_node(f, label_key) or get_node(f, "Y_Y")
    if Xnode is None or Ynode is None:
        raise KeyError("Could not find X and Y in HDF5 file.")
    X = _fix_shape(Xnode[()], expected_cols=4)
    Y = _fix_shape(Ynode[()], expected_cols=8)
    return X, Y

def _extract_from_mat(d, label_key="Y_Y"):
    """
    Extract from dict returned by scipy.io.loadmat.
    Support:
      - struct Dataset with fields X, Y_Y (or label_key)
      - top-level X, Y_Y
    """
    # Remove MATLAB meta-keys
    d2 = {k: v for k, v in d.items() if not k.startswith("__")}
    # struct-like 'Dataset'
    if "Dataset" in d2:
        G = d2["Dataset"]
        # If it's a numpy void (structured array)
        if hasattr(G, "dtype") and G.dtype.names:
            fields = G.dtype.names
            def get_field(name):
                if name in fields:
                    val = G[name]
                    # matlab structs often come as 1x1 arrays
                    val = np.array(val).squeeze()
                    return val
                return None
            X = _fix_shape(get_field("X"), expected_cols=4)
            Yraw = get_field(label_key) if get_field(label_key) is not None else get_field("Y_Y")
            if Yraw is None:
                raise KeyError("Neither label_key nor 'Y_Y' found in Dataset struct.")
            Y = _fix_shape(Yraw, expected_cols=8)
            return X, Y

    # top-level
    X = d2.get("X", None)
    Y = d2.get(label_key, None) or d2.get("Y_Y", None)
    if X is None or Y is None:
        raise KeyError("Could not find X and Y in MAT file (top-level).")
    X = _fix_shape(X, expected_cols=4)
    Y = _fix_shape(Y, expected_cols=8)
    return X, Y

def load_dataset_mat(path, label_key="Y_Y"):
    """
    Load one .mat dataset and return a dict:
      {
        'X': (N,4),
        'Y': (N,8),
        'family': 'GFMI'|'GFLI',
        'ibr': <filename stem>,
        'is_test': True|False
      }
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    lower = path.name.lower()
    family = "GFMI" if "gfmi" in lower else "GFLI"
    is_test = "_test_" in lower
    ibr = path.stem  # e.g., gfli2_impedance_dataset or gfli2_test_impedance_dataset

    # Try HDF5 first
    try:
        with h5py.File(path, "r") as f:
            X, Y = _extract_from_h5(f, label_key=label_key)
            return {"X": X, "Y": Y, "family": family, "ibr": ibr, "is_test": is_test}
    except Exception as e_h5:
        # Fallback to old MAT format
        try:
            d = loadmat(str(path))
            X, Y = _extract_from_mat(d, label_key=label_key)
            return {"X": X, "Y": Y, "family": family, "ibr": ibr, "is_test": is_test}
        except Exception as e_mat:
            raise Exception(f"Error loading {path}: h5py-> {e_h5}; loadmat-> {e_mat}")

# ========= Load TRAIN data =========
loaded_tr, missing_tr = [], []
for fp in TRAIN_FILE_PATHS:
    try:
        d = load_dataset_mat(fp, LABEL_KEY)
        print(f"[TRAIN] Loaded: {fp} -> X:{d['X'].shape}, Y:{d['Y'].shape}, family={d['family']}")
        loaded_tr.append((fp, d))
    except Exception as e:
        print(f"[TRAIN] Error loading {fp}: {e}")
        missing_tr.append((fp, str(e)))

# ========= Load TEST data (evaluation set) =========
loaded_te, missing_te = [], []
for fp in TEST_FILE_PATHS:
    try:
        d = load_dataset_mat(fp, LABEL_KEY)
        print(f"[TEST ] Loaded: {fp} -> X:{d['X'].shape}, Y:{d['Y'].shape}, family={d['family']}")
        loaded_te.append((fp, d))
    except Exception as e:
        print(f"[TEST ] Error loading {fp}: {e}")
        missing_te.append((fp, str(e)))

# ==== Merge TRAIN only ====
X_list, Y_list, fam_list, ibr_list = [], [], [], []

for fp, d in loaded_tr:
    X_list.append(d["X"])
    Y_list.append(d["Y"])
    fam = 0 if d["family"].upper() == "GFMI" else 1
    fam_list.append(np.full((d["X"].shape[0],), fam, dtype=int))
    ibr_list.append(np.array([Path(fp).stem] * d["X"].shape[0], dtype=object))

X_train      = np.vstack(X_list) if X_list else np.empty((0, 4))
Y_train      = np.vstack(Y_list) if Y_list else np.empty((0, 8))
family_train = np.concatenate(fam_list) if fam_list else np.empty((0,), dtype=int)
ibr_train    = np.concatenate(ibr_list) if ibr_list else np.empty((0,), dtype=object)
freq_train   = X_train[:, 3] if X_train.size else np.empty((0,))

print("TRAIN merged shapes:", X_train.shape, Y_train.shape)
if family_train.size:
    print("Family distribution (0=GFMI,1=GFLI):", np.bincount(family_train))

# ==== Keep TEST per-file (no merge) for later evaluation ====

test_sets = {Path(fp).stem: d for fp, d in loaded_te}
print(f"TEST sets available: {list(test_sets.keys())}")

# ==== Use TRAIN (merged) only; prepare grid & convenience aliases ====
import numpy as np
import pandas as pd


Vt, Pt, Qt = X_train[:, 0], X_train[:, 1], X_train[:, 2]
owner_groups = np.array(
    [f"{ibr}|V{v}_P{p}_Q{q}" for ibr, v, p, q in zip(ibr_train, Vt, Pt, Qt)],
    dtype=object
)

print("Training rows (merged TRAIN):", X_train.shape[0])
if family_train.size:
    print("Family distribution (0=GFMI,1=GFLI):", np.bincount(family_train))

# ==== Build bin grid from TRAIN ====
NBINS = int(NBINS)  # ensure int
pos_freq = freq_train[freq_train > 0]
if pos_freq.size == 0:
    raise ValueError("No positive frequencies found in TRAIN.")
lo, hi = pos_freq.min(), pos_freq.max()
edges   = np.logspace(np.log10(lo), np.log10(hi), NBINS + 1)
centers = np.sqrt(edges[:-1] * edges[1:])  # geometric centers

# ==== |Ydd| from TRAIN ====
# Order: [Re(Ydd), Im(Ydd), Re(Ydq), Im(Ydq), Re(Yqd), Im(Yqd), Re(Yqq), Im(Yqq)]
Ydd_mag_train = np.hypot(Y_train[:, 0], Y_train[:, 1])


X_all      = X_train
ibr_labels = ibr_train
family_all = family_train
freq_all   = freq_train
Ydd_mag    = Ydd_mag_train


TRAINING_ON_ALL = True

# ==== Build |Ydd| (TRAIN) + helpers: log-binned median per group ====


Ydd_re_tr = Y_train[:, 0]
Ydd_im_tr = Y_train[:, 1]
Ydd_mag_train = np.hypot(Ydd_re_tr, Ydd_im_tr)

def build_grid(x, nbins: int):
    x = np.asarray(x)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        raise ValueError("No positive frequencies found in TRAIN.")
    nbins = int(nbins)
    lo, hi = x.min(), x.max()
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric centers
    return edges, centers

def reduce_curve(freq, yvals, edges):
    freq = np.asarray(freq); yvals = np.asarray(yvals)
    med = np.full(len(edges) - 1, np.nan, dtype=float)
    for i in range(len(edges) - 1):
        m = (freq >= edges[i]) & (freq < edges[i + 1])
        if np.any(m):
            med[i] = np.median(yvals[m])

    s = pd.Series(med).ffill().bfill()
    return s.values


edges, centers = build_grid(freq_train, NBINS)


Ydd_mag = Ydd_mag_train
freq_all = freq_train

# ==== Build curves (TRAIN merged) ====


curves, owners, truth = [], [], []

Vt, Pt, Qt = X_train[:, 0], X_train[:, 1], X_train[:, 2]
df_tr = pd.DataFrame({'V': Vt, 'P': Pt, 'Q': Qt})
df_tr['IBR'] = ibr_train
df_tr['fam'] = family_train
df_tr['idx'] = np.arange(len(df_tr))

for (v, p, q, ibr), g in df_tr.groupby(['V', 'P', 'Q', 'IBR']):
    idx = g['idx'].values
    if idx.size < 20:
        continue

    vec = reduce_curve(freq_train[idx], Ydd_mag_train[idx], edges)
    curves.append(np.log10(np.maximum(vec, 1e-12)))
    owners.append(f"{ibr}|V{v}_P{p}_Q{q}")
    truth.append(int(g['fam'].iloc[0]))

curves = np.stack(curves, axis=0) if len(curves) else np.empty((0, int(NBINS)))
owners = np.asarray(owners, dtype=object)
truth  = np.asarray(truth, dtype=int)

print("Curves matrix (TRAIN):", curves.shape, "-- owners:", len(owners))

# ==== Auto-select K for KMeans on TRAIN (inertia & silhouette) ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def eval_kmeans_over_k_train(curves_tr, k_min=2, k_max=10, n_init=20, repeats=3, random_state=42):
    """
    curves_tr: (n_train_samples, n_features) -- TRAIN (merged)
    Returns:
      - df_k: inertia and silhouette table by k (avg over repeats)
      - k_best: selected k based on combined ranks (silhouette up, inertia down)
      - scaler_tr: StandardScaler fit on TRAIN for consistent transform
    """
    if curves_tr is None or curves_tr.size == 0:
        raise ValueError("curves_tr is empty. Run 'Build curves' first.")
    n_samples = curves_tr.shape[0]
    if n_samples < max(3, k_min):
        raise ValueError(f"Too few TRAIN samples ({n_samples}) for k_min={k_min}.")


    scaler_tr = StandardScaler()
    Xs_tr = scaler_tr.fit_transform(curves_tr)

    rng = np.random.default_rng(random_state)
    rows = []
    for k in range(k_min, min(k_max, n_samples - 1) + 1):
        sil_list, inertia_list = [], []
        for _ in range(repeats):
            rs = int(rng.integers(0, 10_000))
            km = KMeans(n_clusters=k, n_init=n_init, random_state=rs)
            labels = km.fit_predict(Xs_tr)
            inertia_list.append(km.inertia_)

            try:
                sil = silhouette_score(Xs_tr, labels)
            except Exception:
                sil = np.nan
            sil_list.append(sil)

        rows.append({
            "k": k,
            "inertia_mean": float(np.nanmean(inertia_list)),
            "silhouette_mean": float(np.nanmean(sil_list)),
            "inertia_std": float(np.nanstd(inertia_list)),
            "silhouette_std": float(np.nanstd(sil_list)),
        })

    df_k = pd.DataFrame(rows)


    df_k["rank_sil"] = (-df_k["silhouette_mean"]).rank(method="min")
    df_k["rank_inertia"] = (df_k["inertia_mean"]).rank(method="min")
    df_k["rank_sum"] = df_k["rank_sil"] + df_k["rank_inertia"]

    k_best = int(df_k.loc[df_k["rank_sum"].idxmin(), "k"])
    return df_k, k_best, scaler_tr

def fit_final_kmeans_train(curves_tr, k_best, scaler_tr, n_init=50, random_state=42):
    """
    Fit final KMeans on TRAIN and return:
      - kmeans_tr: fitted KMeans model
      - labels_tr: cluster labels for TRAIN
    """
    Xs_tr = scaler_tr.transform(curves_tr)
    kmeans_tr = KMeans(n_clusters=k_best, n_init=n_init, random_state=random_state)
    labels_tr = kmeans_tr.fit_predict(Xs_tr)
    return kmeans_tr, labels_tr


df_k, k_best, scaler_tr = eval_kmeans_over_k_train(
    curves, k_min=2, k_max=6, n_init=20, repeats=10, random_state=42
)
print("Suggested k_best =", k_best)
display(df_k)


kmeans_tr, labels_tr = fit_final_kmeans_train(curves, k_best, scaler_tr, n_init=50, random_state=42)
print("Cluster counts (train):", np.bincount(labels_tr))


fig, ax1 = plt.subplots(figsize=(6,4))


l1 = ax1.plot(df_k["k"], df_k["inertia_mean"],
              marker="s", color="tab:orange", linestyle="-",
              markersize=9, linewidth=2, markerfacecolor="white",
              markeredgecolor="tab:orange", label="Inertia")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia", color="tab:orange")
ax1.tick_params(axis='y', labelcolor="tab:orange")
ax1.grid(True, ls=":", alpha=0.4)


ax2 = ax1.twinx()
l2 = ax2.plot(df_k["k"], df_k["silhouette_mean"],
              marker="o", color="tab:blue", linestyle="--",
              markersize=9, linewidth=2, markerfacecolor="white",
              markeredgecolor="tab:blue", label="Silhouette")
ax2.set_ylabel("Silhouette", color="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:blue")


ax1.axvline(k_best, ls="--", lw=1.2, color="tab:gray", alpha=0.8)


lines = l1 + l2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.tight_layout()
plt.show()
fig.savefig("kmeans_eval.png", dpi=600, bbox_inches="tight")






X_feat_all = scaler_tr.transform(curves)


pred = kmeans_tr.predict(X_feat_all)

# ----- Metrics vs. family truth (0=GFMI,1=GFLI) -----
def cluster_purity(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    purity_sum = 0
    for c in np.unique(y_pred):
        mask = (y_pred == c)
        if mask.any():
            counts = np.bincount(y_true[mask].astype(int), minlength=2)
            purity_sum += counts.max()
    return purity_sum / n if n > 0 else np.nan


def align_labels_binary(pred, truth):
    acc1 = (pred == truth).mean()
    acc2 = (1 - pred == truth).mean()
    return pred if acc1 >= acc2 else (1 - pred)

pred_aligned = align_labels_binary(pred, truth) if k_best == 2 else pred

purity = cluster_purity(truth, pred_aligned)
ari = adjusted_rand_score(truth, pred)

print(f"[ALL] Purity vs family: {purity:.4f} | ARI: {ari:.4f} | curves = {len(pred)}")



NEW_FILE = "gfli_TEST_cluster_impedance_dataset.mat"


d_new = load_dataset_mat(NEW_FILE, LABEL_KEY)
X_new, Y_new = d_new["X"], d_new["Y"]


freq_new = X_new[:, 3]
Ydd_mag_new = np.hypot(Y_new[:, 0], Y_new[:, 1])  # |Ydd| = sqrt(Re^2 + Im^2)


vec_new = reduce_curve(freq_new, Ydd_mag_new, edges)
curve_new = np.log10(np.maximum(vec_new, 1e-12))[None, :]  # shape (1, NBINS)


try:
    X_feat_new = scaler_tr.transform(curve_new)
    cluster_id = int(kmeans_tr.predict(X_feat_new)[0])
except NameError as e:
    raise RuntimeError(
        "Missing scaler_tr/kmeans_tr. Make sure you fit the model on TRAIN, "
        "e.g.:"
        "  scaler_tr = StandardScaler().fit(curves)\n"
        "  kmeans_tr = KMeans(n_clusters=k_best, random_state=0).fit(scaler_tr.transform(curves))"
    ) from e


dists = kmeans_tr.transform(X_feat_new)[0]

print("Distances to each cluster (Euclidean, scaled):")
for k, d in enumerate(dists):
    print(f"  cluster {k}: {d:.6f}")


order = np.argsort(dists)
print("Nearest to farthest order:")
for k in order:
    print(f"  cluster {int(k)}: {float(dists[k]):.6f}")

best_dist = float(dists[cluster_id])
second_best = float(np.partition(dists, 1)[1]) if len(dists) > 1 else np.nan
margin = second_best - best_dist if np.isfinite(second_best) else np.nan

print(f"[NEW] {NEW_FILE} -> predicted cluster = {cluster_id}")
print(f"Distance to best cluster center = {best_dist:.4f}, margin (2nd-best - best) = {margin:.4f}")



try:

    cluster_to_family = {}
    for c in np.unique(pred):
        mask = (pred == c)
        if mask.any():
            counts = np.bincount(truth[mask].astype(int), minlength=2)
            cluster_to_family[int(c)] = int(np.argmax(counts))
    fam_pred = cluster_to_family.get(cluster_id, None)
    if fam_pred is not None:
        fam_name = "GFMI (0)" if fam_pred == 0 else "GFLI (1)"
        print(f"-> Cluster {cluster_id} majority family: {fam_name}")
except NameError:

    pass


labels_all = pred_aligned if (k_best == 2) else pred
assert curves.shape[0] == len(labels_all) == len(owners), "Mismatch lengths!"


uniq = np.unique(labels_all)
if len(uniq) <= 10:
    cmap = plt.get_cmap('tab10', len(uniq))
elif len(uniq) <= 20:
    cmap = plt.get_cmap('tab20', len(uniq))
else:
    cmap = plt.get_cmap('gist_ncar', len(uniq))
color_map = {int(c): cmap(i) for i, c in enumerate(uniq)}

plt.figure(figsize=(6, 4))
for i in range(curves.shape[0]):
    cid = int(labels_all[i])
    y = np.power(10.0, curves[i])
    plt.plot(centers, y, linewidth=1.5, alpha=0.7, color=color_map[cid])

plt.xscale('log'); plt.yscale('log')
plt.grid(True, which='both', ls=':', alpha=0.4)
plt.xlabel('Frequency (Hz)')
plt.ylabel('|Ydd|')
plt.title(f"|Ydd| Bode lines -- ALL data (k={k_best})")
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=600, bbox_inches="tight")
plt.show()
print('Saved figure ->', OUT_IMG)


labels_all = pred_aligned if (k_best == 2) else pred

assert len(owners) == len(truth) == len(labels_all), "Length mismatch in summary inputs."

summary = pd.DataFrame({
    'owner': owners,                                   # IBR|V..._P..._Q...
    'family_true': np.where(truth == 0, 'GFMI', 'GFLI'),
    'family_id': truth.astype(int),
    'cluster': labels_all.astype(int),
})

summary.to_csv(OUT_SUMMARY, index=False)
print('Saved summary ->', OUT_SUMMARY)
summary.head()


from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------- 0) Config ----------

target_cols = slice(0, 8)


DEFAULT_HIDDEN = (32, 32)


HIDDEN_BY_CLUSTER = {
    0: (4, 8),     # GFMI
    1: (32, 32),   # GFLI
    2: (32, 32),   # GFLI
}

max_iter = 1200
random_state = 42

def make_regressor(hidden=DEFAULT_HIDDEN, maxit=max_iter, rs=random_state):
    """
    Multi-output regression pipeline per cluster:
      - Scale X with StandardScaler
      - MLPRegressor multi-output (Y has 8 columns)
    """
    return Pipeline([
        ("x_scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            max_iter=maxit,
            random_state=rs,
        ))
    ])

def get_hidden_for_cluster(c: int):
    """Return a suitable hidden-layer layout for cluster c."""
    return HIDDEN_BY_CLUSTER.get(int(c), DEFAULT_HIDDEN)



labels_train = labels_tr


owner_to_cluster_train = {own: int(lbl) for own, lbl in zip(owners, labels_train)}


Vt, Pt, Qt = X_train[:, 0], X_train[:, 1], X_train[:, 2]
owner_key_train = np.array(
    [f"{ibr}|V{v}_P{p}_Q{q}" for ibr, v, p, q in zip(ibr_train, Vt, Pt, Qt)],
    dtype=object
)


cluster_id_train = np.array([owner_to_cluster_train.get(ok, -1) for ok in owner_key_train], dtype=int)
valid_train_mask = (cluster_id_train >= 0)


clusters = np.unique(labels_train)
train_idx_by_cluster = {c: np.where((cluster_id_train == c) & valid_train_mask)[0] for c in clusters}

print("TRAIN rows by cluster:")
for c in clusters:
    print(f"  Cluster {c}: {train_idx_by_cluster[c].size} rows")




test_idx_by_cluster = {c: {} for c in clusters}

for test_name, d in test_sets.items():
    X_te = d["X"]
    Y_te = d["Y"]
    ibr_name = d["ibr"]


    Vte, Pte, Qte, Fte = X_te[:, 0], X_te[:, 1], X_te[:, 2], X_te[:, 3]
    df_te = pd.DataFrame({
        "V": Vte,
        "P": Pte,
        "Q": Qte,
        "idx": np.arange(X_te.shape[0])
    })



    Ydd_mag_test = np.hypot(Y_te[:, 0], Y_te[:, 1])

    owners_te, curves_te = [], []
    for (v, p, q), g in df_te.groupby(["V", "P", "Q"]):
        idx = g["idx"].values
        if idx.size < 20:
            continue
        vec = reduce_curve(Fte[idx], Ydd_mag_test[idx], edges)
        curves_te.append(np.log10(np.maximum(vec, 1e-12)))
        owners_te.append(f"{ibr_name}|V{v}_P{p}_Q{q}")


    if len(curves_te) > 0:
        curves_test_mat = np.stack(curves_te, axis=0)
        Xs_te_curves = scaler_tr.transform(curves_test_mat)
        labels_te_owners = kmeans_tr.predict(Xs_te_curves)
        owner_to_cluster_test = {own: int(lbl) for own, lbl in zip(owners_te, labels_te_owners)}
    else:
        owner_to_cluster_test = {}


    owner_key_test = np.array(
        [f"{ibr_name}|V{v}_P{p}_Q{q}" for v, p, q in zip(Vte, Pte, Qte)],
        dtype=object
    )
    cluster_id_test = np.array([owner_to_cluster_test.get(ok, -1) for ok in owner_key_test], dtype=int)
    valid_test_mask = (cluster_id_test >= 0)


    for c in clusters:
        idxs = np.where((cluster_id_test == c) & valid_test_mask)[0]
        if idxs.size > 0:
            test_idx_by_cluster[c][test_name] = idxs


print("\nTEST rows by cluster (per file):")
for c in clusters:
    total_c = sum(len(v) for v in test_idx_by_cluster[c].values())
    detail = ", ".join([f"{k}:{len(v)}" for k, v in test_idx_by_cluster[c].items()])
    print(f"  Cluster {c}: total {total_c} rows | {detail if detail else '--'}")

from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

# --------- 4) Train 1 FNN/cluster ----------
models = {}
stats_rows = []

for c in clusters:
    tr_idx = train_idx_by_cluster.get(c, np.array([], dtype=int))
    if tr_idx.size == 0:
        print(f"[WARN] Cluster {c}: no TRAIN rows, skip.")
        continue

    X_tr_c = X_train[tr_idx]
    Y_tr_c = Y_train[tr_idx][:, target_cols]


    hidden_c = get_hidden_for_cluster(c)

    pipe = TransformedTargetRegressor(
        regressor=make_regressor(hidden=hidden_c),
        transformer=StandardScaler(with_mean=True, with_std=True)
    )
    pipe.fit(X_tr_c, Y_tr_c)
    models[c] = pipe


    Y_tr_hat = pipe.predict(X_tr_c)
    mae_tr = mean_absolute_error(Y_tr_c, Y_tr_hat)
    rmse_tr = mean_squared_error(Y_tr_c, Y_tr_hat)

    stats_rows.append({
        "cluster": int(c), "split": "train", "test_name": "",
        "n": int(tr_idx.size), "MAE": float(mae_tr), "RMSE": float(rmse_tr)
    })


    te_dict = test_idx_by_cluster.get(c, {})
    X_blocks, Y_blocks, n_total = [], [], 0

    for test_name, idxs in te_dict.items():
        X_te_c = test_sets[test_name]["X"][idxs]
        Y_te_c = test_sets[test_name]["Y"][idxs][:, target_cols]

        Y_te_hat = pipe.predict(X_te_c)
        mae_te = mean_absolute_error(Y_te_c, Y_te_hat)
        rmse_te = mean_squared_error(Y_te_c, Y_te_hat)

        stats_rows.append({
            "cluster": int(c), "split": "test", "test_name": test_name,
            "n": int(idxs.size), "MAE": float(mae_te), "RMSE": float(rmse_te)
        })

        X_blocks.append(X_te_c)
        Y_blocks.append(Y_te_c)
        n_total += idxs.size

    if n_total > 0:
        X_te_all = np.vstack(X_blocks)
        Y_te_all = np.vstack(Y_blocks)
        Y_te_all_hat = pipe.predict(X_te_all)
        mae_te_all = mean_absolute_error(Y_te_all, Y_te_all_hat)
        rmse_te_all = mean_squared_error(Y_te_all, Y_te_all_hat)

        stats_rows.append({
            "cluster": int(c), "split": "test_all", "test_name": "ALL",
            "n": int(n_total), "MAE": float(mae_te_all), "RMSE": float(rmse_te_all)
        })


stats_df = pd.DataFrame(stats_rows).sort_values(["cluster", "split", "test_name"]).reset_index(drop=True)
display(stats_df)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


Y_te_true_all_list = []
Y_te_pred_all_list = []

for c, mdl in models.items():
    te_dict = test_idx_by_cluster.get(c, {})
    for test_name, idxs in te_dict.items():
        if idxs.size == 0:
            continue
        X_te = test_sets[test_name]["X"][idxs]
        Y_te = test_sets[test_name]["Y"][idxs][:, target_cols]

        Y_hat = mdl.predict(X_te)

        Y_te_true_all_list.append(Y_te)
        Y_te_pred_all_list.append(Y_hat)


if len(Y_te_true_all_list) == 0:
    raise RuntimeError("No valid TEST rows to plot scatter (test_idx_by_cluster is empty).")

Y_te_true_all = np.vstack(Y_te_true_all_list)
Y_te_pred_all = np.vstack(Y_te_pred_all_list)


names = [
    "Re(Ydd)", "Im(Ydd)",
    "Re(Ydq)", "Im(Ydq)",
    "Re(Yqd)", "Im(Yqd)",
    "Re(Yqq)", "Im(Yqq)",
]
n_out = Y_te_true_all.shape[1]
assert n_out == len(names) == 8, f"Output column count ({n_out}) does not match 8."

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
axes = axes.ravel()

for j in range(n_out):
    y_true = Y_te_true_all[:, j]
    y_pred = Y_te_pred_all[:, j]

    ax = axes[j]
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)

    # y = x
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], lw=1.3)


    if y_true.size >= 2 and np.std(y_true) > 0:
        coef = np.polyfit(y_true, y_pred, 1)
        reg_line = np.poly1d(coef)
        ax.plot([lo, hi], reg_line([lo, hi]), lw=1.3, linestyle="--")
        r2 = r2_score(y_true, y_pred)
        ax.set_title(f"{names[j]}\nSlope={coef[0]:.3f}, Intercept={coef[1]:.3f}, R2={r2:.4f}")
    else:
        ax.set_title(f"{names[j]}")

    ax.set_xlabel("True")
    ax.set_ylabel("Pred")
    ax.grid(True, ls=":", alpha=0.4)

plt.tight_layout()
plt.show()


from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


N_OWNERS_PER_CLUSTER = 1


components = [
    ("Ydd", 0, 1),
    ("Ydq", 2, 3),
    ("Yqd", 4, 5),
    ("Yqq", 6, 7),
]

for c, mdl in models.items():

    te_dict = test_idx_by_cluster.get(c, {})
    if not te_dict:
        print(f"[INFO] Cluster {c}: no TEST rows.")
        continue


    owners_selected = []  # list of tuples: (owner_key, test_name, idxs_local)
    for test_name, idxs in te_dict.items():
        if len(owners_selected) >= N_OWNERS_PER_CLUSTER:
            break
        X_te = test_sets[test_name]["X"][idxs]
        Y_te = test_sets[test_name]["Y"][idxs]
        ibr_name = test_sets[test_name]["ibr"]


        Vte, Pte, Qte = X_te[:, 0], X_te[:, 1], X_te[:, 2]
        df_te = pd.DataFrame({
            "V": Vte, "P": Pte, "Q": Qte,
            "idx_local": np.arange(X_te.shape[0])
        })

        for (v, p, q), g in df_te.groupby(["V", "P", "Q"]):
            if len(owners_selected) >= N_OWNERS_PER_CLUSTER:
                break
            local_idx = g["idx_local"].values
            if local_idx.size < 20:
                continue
            owner_key = f"{ibr_name}|V{v}_P{p}_Q{q}"
            owners_selected.append((owner_key, test_name, local_idx))

    if not owners_selected:
        print(f"[INFO] Cluster {c}: no owner with enough points (>=20) to plot.")
        continue


    fig_re, axes_re = plt.subplots(2, 2, figsize=(5, 4),
                               gridspec_kw={'hspace': 0.7, 'wspace': 0.3})
    axes_re = axes_re.ravel()

    fig_im, axes_im = plt.subplots(2, 2, figsize=(5, 4),
                               gridspec_kw={'hspace': 0.7, 'wspace': 0.3})
    axes_im = axes_im.ravel()

    legend_proxies = [
    Line2D([0], [0], color='tab:blue', lw=2.5, ls='-',  label='TRUE'),
    Line2D([0], [0], color='tab:orange', lw=2.5, ls='--', label='PRED'),
    ]

    for owner_key, test_name, local_idx in owners_selected:
        X_te_block = test_sets[test_name]["X"][te_dict[test_name]]
        Y_te_block = test_sets[test_name]["Y"][te_dict[test_name]]


        X_owner = X_te_block[local_idx, :]
        Y_owner_true = Y_te_block[local_idx, :]

        f = X_owner[:, 3]
        Y_owner_pred = mdl.predict(X_owner)
        short_owner = owner_key.split('|', 1)[-1].replace('_', ' ').replace('=', '')
        if 'gfmi' in owner_key.lower(): tag = 'GFMI'
        elif 'gfli' in owner_key.lower(): tag = 'GFLI'
        else: tag = owner_key.split('|',1)[0]
        owner_tag = f"{tag} | {short_owner}"

        for comp_i, (name, re_i, im_i) in enumerate(components):
            # --- Re ---
            re_true = Y_owner_true[:, re_i]
            re_pred = Y_owner_pred[:, re_i]
            curve_re_true = reduce_curve(f, re_true, edges)
            curve_re_pred = reduce_curve(f, re_pred, edges)

            axr = axes_re[comp_i]
            axr.semilogx(centers, curve_re_true, linewidth=3.0, alpha=0.9)
            axr.semilogx(centers, curve_re_pred, linewidth=3.0, alpha=0.9, linestyle="--")
            axr.grid(True, which='both', ls=':', alpha=0.4)
            axr.set_title(f"Re({name})")
            axr.set_xlabel('Frequency (Hz)')
            axr.set_ylabel('')
            axr.legend(handles=legend_proxies, loc="upper right",
                   fontsize=8, framealpha=0.9, borderpad=0.4, handlelength=2.5)
            axr.text(0.02, 0.98, owner_tag, transform=axr.transAxes,
                 va='top', ha='left', fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8))

            # --- Im ---
            im_true = Y_owner_true[:, im_i]
            im_pred = Y_owner_pred[:, im_i]
            curve_im_true = reduce_curve(f, im_true, edges)
            curve_im_pred = reduce_curve(f, im_pred, edges)

            axi = axes_im[comp_i]
            axi.semilogx(centers, curve_im_true, linewidth=3.0, alpha=0.9)
            axi.semilogx(centers, curve_im_pred, linewidth=3.0, alpha=0.9, linestyle="--")
            axi.grid(True, which='both', ls=':', alpha=0.4)
            axi.set_title(f"Im({name})")
            axi.set_xlabel('Frequency (Hz)')
            axi.set_ylabel('')
            axi.legend(handles=legend_proxies, loc="upper right",
                   fontsize=8, framealpha=0.9, borderpad=0.4, handlelength=2.5)

            axi.text(0.02, 0.98, owner_tag, transform=axi.transAxes,
                 va='top', ha='left', fontsize=7,
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.8))


    fig_re.savefig(f"real_parts_cluster{c}.png", dpi=600, bbox_inches="tight", facecolor="white")
    fig_im.savefig(f"imag_parts_cluster{c}.png", dpi=600, bbox_inches="tight", facecolor="white")
plt.show()


# - Input true file: gfmi_TEST_impedance_dataset.mat



import matplotlib.cm as cm
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path

# --------- Config ----------
TRUE_FILE = "gfli_TEST_impedance_dataset.mat"
CLUSTER_FOR_NEW = int(cluster_id)
FIX_V = None
FIX_Q = None   # None = auto
SWEEP_NBINS = 24
DB_EPS = 1e-12

# --------- Helpers ----------
def mag_db_real(re, eps=DB_EPS):
    """20*log10(|Re(.)| + eps) to drop the imaginary part as requested."""
    return 20.0 * np.log10(np.abs(re) + eps)

def choose_mode(vals):
    s = pd.Series(vals).round(9).value_counts()
    return float(s.index[0]) if len(s) else float(np.median(vals) if len(vals) else 0.0)

def build_lin_edges(x, nbins):
    x = np.asarray(x)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        hi = lo + 1.0
    return np.linspace(lo, hi, int(nbins) + 1)

def grid_median(freq, sweep, z, f_edges, s_edges):
    """Binning by f (log edges from TRAIN) and sweep (linear). Median per cell; ffill/bfill edges."""
    Fmat = np.full((len(s_edges)-1, len(f_edges)-1), np.nan, float)
    for i in range(len(s_edges)-1):
        ms = (sweep >= s_edges[i]) & (sweep < s_edges[i+1])
        if not np.any(ms):
            continue
        f_sub, z_sub = freq[ms], z[ms]
        for j in range(len(f_edges)-1):
            mf = (f_sub >= f_edges[j]) & (f_sub < f_edges[j+1])
            if np.any(mf):
                Fmat[i, j] = np.median(z_sub[mf])

    Fmat = pd.DataFrame(Fmat).ffill(axis=0).bfill(axis=0).ffill(axis=1).bfill(axis=1).values
    f_cent = np.sqrt(f_edges[:-1] * f_edges[1:])   # log centers
    s_cent = 0.5 * (s_edges[:-1] + s_edges[1:])
    return f_cent, s_cent, Fmat

def log_ticks(minv, maxv):
    a, b = np.floor(np.log10(minv)), np.ceil(np.log10(maxv))
    vals = 10.0 ** np.arange(a, b+1)
    return np.log10(vals), [f"$10^{{{int(x)}}}$" for x in range(int(a), int(b)+1)]


def _shared_norm(surfaces, mode="mag"):
    """Shared normalization for the row of four subplots."""
    vals = []
    for k in ["Ydd", "Ydq", "Yqd", "Yqq"]:
        if k in surfaces and surfaces[k] is not None:
            Z = surfaces[k][2]
            if Z.size:
                vals.append(Z[np.isfinite(Z)])
    if not vals:
        return None
    vmin = min(v.min() for v in vals)
    vmax = max(v.max() for v in vals)
    if mode == "err":
        v = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)
    return Normalize(vmin=vmin, vmax=vmax)

def plot_surface_row(axarr, surfaces, row_title,
                     zlabel="Magnitude (dB)", ylabel="P (pu)", mode="mag"):
    """
    mode='mag' uses a sequential colormap (viridis); mode='err' uses a diverging colormap (coolwarm, center=0).
    """
    comps = ["Ydd", "Ydq", "Yqd", "Yqq"]
    norm = _shared_norm(surfaces, mode=mode)
    cmap = mpl.colormaps['viridis'] if mode != 'err' else mpl.colormaps['coolwarm']
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    for j, name in enumerate(comps):
        ax = axarr[j]
        if name not in surfaces or surfaces[name] is None:
            ax.set_title(f"{name} (no data)", fontsize=10); continue

        f_cent, s_cent, Z = surfaces[name]
        Xg, Yg = np.meshgrid(np.log10(f_cent), s_cent)


        facecolors = cmap(norm(Z))
        ax.plot_surface(Xg, Yg, Z,
                        facecolors=facecolors,
                        rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, alpha=1.0)


        ax.plot_wireframe(Xg, Yg, Z, rstride=3, cstride=3, color="k", linewidth=0.15, alpha=0.25)


        xtick, xtlbl = log_ticks(f_cent.min(), f_cent.max())
        ax.set_xticks(xtick); ax.set_xticklabels(xtlbl, fontsize=8)
        ax.set_xlabel('Frequency (Hz)', labelpad=4)
        ax.set_ylabel(ylabel, labelpad=4)
        ax.set_zlabel(zlabel, labelpad=6)
        ax.set_title(name, fontsize=14)
        ax.view_init(elev=25, azim=-60)
        ax.set_box_aspect((1.2, 1.0, 0.6))


    # fig = axarr[0].figure
    # cbar = fig.colorbar(mappable, ax=axarr.ravel().tolist(),

    # cbar.set_label(zlabel, rotation=90)
    # fig.suptitle(row_title, y=0.98, fontsize=11)


# --------- Load true file & filter V,Q ----------
d_true = load_dataset_mat(TRUE_FILE, LABEL_KEY)
X_true, Y_true = d_true["X"], d_true["Y"]
V, P, Q, F = X_true[:,0], X_true[:,1], X_true[:,2], X_true[:,3]

v_fix = choose_mode(V) if FIX_V is None else float(FIX_V)
q_fix = choose_mode(Q) if FIX_Q is None else float(FIX_Q)
mask_vq = np.isclose(V, v_fix, atol=1e-9) & np.isclose(Q, q_fix, atol=1e-9)


if mask_vq.sum() < 80:
    v_counts = pd.Series(V).round(9).value_counts()
    q_counts = pd.Series(Q).round(9).value_counts()
    for vv in v_counts.index:
        for qq in q_counts.index:
            m2 = np.isclose(V, float(vv), atol=1e-9) & np.isclose(Q, float(qq), atol=1e-9)
            if m2.sum() >= 80:
                v_fix, q_fix = float(vv), float(qq)
                mask_vq = m2
                break
        if mask_vq.sum() >= 80:
            break

X_sub   = X_true[mask_vq]
Y_sub_T = Y_true[mask_vq]     # ground truth
P_sub   = P[mask_vq]
F_sub   = F[mask_vq]

if X_sub.size == 0:
    raise RuntimeError("No data found with fixed V,Q in true file. Set FIX_V, FIX_Q manually.")

print(f"[INFO] Using V={v_fix:g}, Q={q_fix:g} | rows={X_sub.shape[0]} | cluster={CLUSTER_FOR_NEW}")

# --------- Predict with pretrained model of the detected cluster ----------
if CLUSTER_FOR_NEW not in models:
    raise KeyError(f"No model for cluster {CLUSTER_FOR_NEW}. Check per-cluster training step.")

Y_sub_P = models[CLUSTER_FOR_NEW].predict(X_sub)

# --------- Build surfaces (median bins) for True / Pred / Error (Re-only, in dB) ----------

def build_surfaces_real_re_db(Ymat, F_vec, P_vec, f_edges, p_edges):
    comps = {
        "Ydd": mag_db_real(Ymat[:, 0]),
        "Ydq": mag_db_real(Ymat[:, 2]),
        "Yqd": mag_db_real(Ymat[:, 4]),
        "Yqq": mag_db_real(Ymat[:, 6]),
    }
    out = {}
    for name, z in comps.items():
        f_cent, p_cent, Z = grid_median(F_vec, P_vec, z, f_edges=f_edges, s_edges=p_edges)
        out[name] = (f_cent, p_cent, Z)
    return out

p_edges = build_lin_edges(P_sub, SWEEP_NBINS)
f_edges = edges

sur_true = build_surfaces_real_re_db(Y_sub_T, F_sub, P_sub, f_edges, p_edges)
sur_pred = build_surfaces_real_re_db(Y_sub_P, F_sub, P_sub, f_edges, p_edges)

# Error (dB) = Pred_dB - True_dB
sur_err = {}
for key in sur_true.keys():
    f_cent_t, p_cent_t, Zt = sur_true[key]
    f_cent_p, p_cent_p, Zp = sur_pred[key]

    sur_err[key] = (f_cent_t, p_cent_t, Zp - Zt)


fig1, axes1 = plt.subplots(1, 4, figsize=(15, 4.6), subplot_kw={"projection": "3d"})
plot_surface_row(axes1, sur_true,
                 row_title=f"TRUE * |Re(Y)| (dB) * V={v_fix:g}, Q={q_fix:g} * Cluster {CLUSTER_FOR_NEW}",
                 zlabel="|Re(Y)| (dB)", ylabel="P (pu)", mode="mag")
fig1.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.14, wspace=0.10)
plt.show()

fig2, axes2 = plt.subplots(1, 4, figsize=(15, 4.6), subplot_kw={"projection": "3d"})
plot_surface_row(axes2, sur_pred,
                 row_title=f"PRED * |Re(Y)| (dB) * V={v_fix:g}, Q={q_fix:g} * Cluster {CLUSTER_FOR_NEW}",
                 zlabel="|Re(Y)| (dB)", ylabel="P (pu)", mode="mag")
fig2.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.14, wspace=0.10)
plt.show()

fig3, axes3 = plt.subplots(1, 4, figsize=(15, 4.6), subplot_kw={"projection": "3d"})
plot_surface_row(axes3, sur_err,
                 row_title=f"ERROR (Pred - True) * |Re(Y)| (dB) * V={v_fix:g}, Q={q_fix:g} * Cluster {CLUSTER_FOR_NEW}",
                 zlabel="Delta|Re(Y)| (dB)", ylabel="P (pu)", mode="err")
fig3.subplots_adjust(left=0.05, right=0.97, top=0.88, bottom=0.14, wspace=0.10)



outdir = Path("exports_plots")
outdir.mkdir(exist_ok=True)
tag = f"V{v_fix:g}_Q{q_fix:g}_cluster{CLUSTER_FOR_NEW}"


PNG_DPI = 600
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.02, facecolor="white")

# TRUE
fig1.savefig(outdir / f"TRUE_ReY_{tag}.png", dpi=PNG_DPI, **SAVE_KW)
fig1.savefig(outdir / f"TRUE_ReY_{tag}.pdf", **SAVE_KW)

# PRED
fig2.savefig(outdir / f"PRED_ReY_{tag}.png", dpi=PNG_DPI, **SAVE_KW)
fig2.savefig(outdir / f"PRED_ReY_{tag}.pdf", **SAVE_KW)

# ERROR
fig3.savefig(outdir / f"ERROR_ReY_{tag}.png", dpi=PNG_DPI, **SAVE_KW)
fig3.savefig(outdir / f"ERROR_ReY_{tag}.pdf", **SAVE_KW)
plt.show()
