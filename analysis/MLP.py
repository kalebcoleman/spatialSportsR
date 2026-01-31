"""
Shot Quality (xFG) v1 — PyTorch Fully Connected Neural Network (MLP)

Reads from SQLite table: shots_model_modern
Trains an MLP with embeddings for categorical shot descriptors.
Splits by season (train/val/test) to avoid leakage.
Prints a sample of predictions and writes test-season xFG predictions back to SQLite.

Update SQLITE_PATH if needed.
"""

import os
import sqlite3
import time
import pandas as pd
import numpy as np

from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Config
# -----------------------
SQLITE_PATH = "../data/parsed/nba.sqlite"  # <-- CHANGE THIS to your sqlite file path
TABLE = "shots_model_modern"
PLAYER_TABLE = "shots_context_mat_modern"
USE_PLAYER_ID = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4096
EPOCHS = 25
LR = 1e-3
SEED = 42
HIDDEN_SIZES = (256, 128, 64)
DROPOUT = 0.3
LOG_EVERY = 200  # batches; set to 0 to disable batch logs

# Season split (edit if needed)
TRAIN_SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2016, 2024)]  # 2016-17..2023-24
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"

# How many predictions to print from test set
N_PRINT_PRED = 25

np.random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------
# Load data
# -----------------------
def _resolve_sqlite_path(sqlite_path: str) -> str:
    if os.path.isabs(sqlite_path):
        return sqlite_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, sqlite_path))


def load_table(sqlite_path: str, table: str) -> pd.DataFrame:
    resolved_path = _resolve_sqlite_path(sqlite_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"SQLite file not found: {resolved_path}\n"
            "Update SQLITE_PATH to a valid file."
        )
    con = sqlite3.connect(resolved_path)
    try:
        df_ = pd.read_sql_query(f"SELECT * FROM {table}", con)
    except Exception as exc:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [r[0] for r in cur.fetchall()]
        raise RuntimeError(
            f"Failed to read table '{table}' from {resolved_path}.\n"
            f"Available tables: {tables}"
        ) from exc
    finally:
        con.close()
    return df_


df = load_table(SQLITE_PATH, TABLE)
if USE_PLAYER_ID:
    resolved_path = _resolve_sqlite_path(SQLITE_PATH)
    con = sqlite3.connect(resolved_path)
    try:
        player_df = pd.read_sql_query(
            f"SELECT GAME_ID, event_num, PLAYER_ID FROM {PLAYER_TABLE}", con
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load PLAYER_ID from {PLAYER_TABLE}. "
            "Set USE_PLAYER_ID = False to skip."
        ) from exc
    finally:
        con.close()
    player_df["GAME_ID"] = player_df["GAME_ID"].astype(str)
    player_df["event_num"] = player_df["event_num"].astype(int)
    df = df.merge(player_df, on=["GAME_ID", "event_num"], how="left")

# Basic cleanup / types
required_cols = [
    "GAME_ID",
    "event_num",
    "season",
    "season_type",
    "SHOT_MADE_FLAG",
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "seconds_left_in_period",
    "PERIOD",
    "is_three",
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_TYPE",
]
if USE_PLAYER_ID:
    required_cols.append("PLAYER_ID")
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns in {TABLE}: {missing}\n"
        f"Run: PRAGMA table_info({TABLE}); and confirm column names."
    )

df = df.dropna(
    subset=[
        "SHOT_MADE_FLAG",
        "LOC_X",
        "LOC_Y",
        "SHOT_DISTANCE",
        "seconds_left_in_period",
        "PERIOD",
    ]
)
df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)
df["PERIOD"] = df["PERIOD"].astype(int)
df["is_three"] = df["is_three"].astype(int)
df["event_num"] = df["event_num"].astype(int)
df["GAME_ID"] = df["GAME_ID"].astype(str)
df["season"] = df["season"].astype(str)
df["season_type"] = df["season_type"].astype(str)
if USE_PLAYER_ID and "PLAYER_ID" in df.columns:
    df["PLAYER_ID"] = df["PLAYER_ID"].astype(str)

# Keep only seasons we intend to use (prevents surprises)
keep_seasons = set(TRAIN_SEASONS + [VAL_SEASON, TEST_SEASON])
df = df[df["season"].isin(keep_seasons)].copy()

if df.empty:
    raise ValueError(
        "After filtering to TRAIN/VAL/TEST seasons, the dataset is empty.\n"
        "Run: SELECT DISTINCT season FROM shots_model_modern ORDER BY season;"
    )


# -----------------------
# Feature setup
# -----------------------
cat_cols = [
    "SHOT_ZONE_BASIC",
    "SHOT_ZONE_AREA",
    "SHOT_ZONE_RANGE",
    "SHOT_TYPE",
    "season_type",
]
if USE_PLAYER_ID:
    cat_cols.append("PLAYER_ID")

# Basic feature engineering for more signal
df["abs_loc_x"] = df["LOC_X"].abs()
df["abs_loc_y"] = df["LOC_Y"].abs()
df["shot_angle"] = np.arctan2(df["LOC_Y"].values, df["LOC_X"].values)
df["dist_sq"] = df["SHOT_DISTANCE"].values ** 2
period_len = np.where(df["PERIOD"].values <= 4, 12 * 60, 5 * 60)
df["seconds_left_frac"] = df["seconds_left_in_period"].values / period_len
df["is_ot"] = (df["PERIOD"].values > 4).astype(int)
df["is_corner_three"] = (
    (df["is_three"].values == 1)
    & (df["abs_loc_x"].values > 220)
    & (df["abs_loc_y"].values < 100)
).astype(int)

num_cols = [
    "LOC_X",
    "LOC_Y",
    "abs_loc_x",
    "abs_loc_y",
    "shot_angle",
    "SHOT_DISTANCE",
    "dist_sq",
    "seconds_left_in_period",
    "seconds_left_frac",
    "PERIOD",
    "is_ot",
    "is_three",
    "is_corner_three",
]

# Fill missing categories safely
for c in cat_cols:
    df[c] = df[c].fillna("UNK").astype(str)

# Map categories -> integer ids (0..n-1), with UNK handling
cat_maps = {}
cat_cardinalities = {}
for c in cat_cols:
    cats = df[c].unique().tolist()
    if "UNK" not in cats:
        cats = ["UNK"] + cats
    cat_maps[c] = {v: i for i, v in enumerate(cats)}
    cat_cardinalities[c] = len(cats)
    df[c + "_id"] = df[c].map(cat_maps[c]).fillna(0).astype(int)

cat_id_cols = [c + "_id" for c in cat_cols]

# Split by season (no leakage)
train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
val_df = df[df["season"] == VAL_SEASON].copy()
test_df = df[df["season"] == TEST_SEASON].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError(
        f"One of your splits is empty.\n"
        f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}\n"
        "Fix TRAIN_SEASONS / VAL_SEASON / TEST_SEASON to match your data."
    )

# Scale numeric features using train only
scaler = StandardScaler()
train_num = scaler.fit_transform(train_df[num_cols].values)
val_num = scaler.transform(val_df[num_cols].values)
test_num = scaler.transform(test_df[num_cols].values)


# -----------------------
# Dataset / Loader
# -----------------------
class ShotsDataset(Dataset):
    def __init__(self, num_x, cat_x, y, keys=None):
        self.num_x = torch.tensor(num_x, dtype=torch.float32)
        self.cat_x = torch.tensor(cat_x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        self.keys = keys

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        if self.keys is None:
            return self.num_x[idx], self.cat_x[idx], self.y[idx]
        return self.num_x[idx], self.cat_x[idx], self.y[idx], self.keys[idx]


def make_keys(d: pd.DataFrame):
    return list(zip(d["GAME_ID"].astype(str).values, d["event_num"].astype(int).values))


train_ds = ShotsDataset(
    train_num, train_df[cat_id_cols].values, train_df["SHOT_MADE_FLAG"].values
)
val_ds = ShotsDataset(
    val_num, val_df[cat_id_cols].values, val_df["SHOT_MADE_FLAG"].values
)
test_ds = ShotsDataset(
    test_num,
    test_df[cat_id_cols].values,
    test_df["SHOT_MADE_FLAG"].values,
    keys=make_keys(test_df),
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -----------------------
# Model: Embeddings + MLP
# -----------------------
def emb_dim(card):
    # small heuristic for embedding size
    return min(50, max(4, int(round(card**0.25 * 8))))


class ShotMLP(nn.Module):
    def __init__(self, cat_cardinalities, n_num, hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.cat_cols = list(cat_cardinalities.keys())
        self.embs = nn.ModuleList(
            [
                nn.Embedding(cat_cardinalities[c], emb_dim(cat_cardinalities[c]))
                for c in self.cat_cols
            ]
        )
        emb_total = sum(e.embedding_dim for e in self.embs)

        layers = []
        in_dim = emb_total + n_num
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers += [nn.Linear(in_dim, 1)]  # logits
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        # x_cat shape: [B, num_cat]
        embs = []
        for i, e in enumerate(self.embs):
            embs.append(e(x_cat[:, i]))
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x)


model = ShotMLP(
    cat_cardinalities, n_num=len(num_cols), hidden=HIDDEN_SIZES, dropout=DROPOUT
).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
crit = nn.BCEWithLogitsLoss()

print(
    f"Device={DEVICE} | train={len(train_df)} | val={len(val_df)} | test={len(test_df)} | "
    f"batch_size={BATCH_SIZE} | epochs={EPOCHS}"
)


# -----------------------
# Train / Eval loops
# -----------------------
@torch.no_grad()
def eval_loader(dl):
    model.eval()
    all_y, all_p = [], []
    for batch in dl:
        if len(batch) == 4:
            x_num, x_cat, y, _ = batch
        else:
            x_num, x_cat, y = batch
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat).cpu().numpy().reshape(-1)
        p = 1 / (1 + np.exp(-logits))
        all_y.append(y.numpy().reshape(-1))
        all_p.append(p)
    y_true = np.concatenate(all_y)
    y_prob = np.concatenate(all_p)

    out = {
        "logloss": float(log_loss(y_true, y_prob)),
        "auc": (
            float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else float("nan")
        ),
        "acc@0.5": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
    }
    return out


for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    running = 0.0
    n = 0
    for batch_idx, (x_num, x_cat, y) in enumerate(train_dl, start=1):
        x_num, x_cat, y = x_num.to(DEVICE), x_cat.to(DEVICE), y.to(DEVICE)
        opt.zero_grad(set_to_none=True)
        logits = model(x_num, x_cat)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        running += loss.item() * y.shape[0]
        n += y.shape[0]
        if LOG_EVERY and (batch_idx % LOG_EVERY == 0 or batch_idx == len(train_dl)):
            avg = running / n
            print(
                f"Epoch {epoch:02d} | batch {batch_idx:04d}/{len(train_dl):04d} | "
                f"train_loss={avg:.4f}"
            )

    train_loss = running / n
    val_metrics = eval_loader(val_dl)
    dt = time.time() - t0
    print(
        f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
        f"val_logloss={val_metrics['logloss']:.4f} | val_auc={val_metrics['auc']:.4f} | "
        f"val_acc={val_metrics['acc@0.5']:.4f} | time={dt:.1f}s"
    )

test_metrics = eval_loader(test_dl)
print("TEST:", test_metrics)


# -----------------------
# Predict + print sample + write to SQLite
# -----------------------
@torch.no_grad()
def predict_with_keys_and_print(dl, n_print=20):
    model.eval()
    rows = []
    printed = 0

    for x_num, x_cat, y, keys in dl:
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        logits = model(x_num, x_cat).cpu().numpy().reshape(-1)
        probs = 1 / (1 + np.exp(-logits))
        y_true = y.numpy().reshape(-1)

        if isinstance(keys, (tuple, list)) and len(keys) == 2:
            game_ids, event_nums = keys
            iter_keys = zip(game_ids, event_nums)
        else:
            iter_keys = keys

        for (game_id, event_num), p, yt in zip(iter_keys, probs, y_true):
            rows.append((str(game_id), int(event_num), float(p)))

            if printed < n_print:
                print(
                    f"GAME_ID={game_id} | event_num={event_num} | "
                    f"xFG={p:.3f} | made={int(yt)}"
                )
                printed += 1

    return rows


pred_rows = predict_with_keys_and_print(test_dl, n_print=N_PRINT_PRED)

# quick printed summary stats
pred_df = pd.DataFrame(pred_rows, columns=["GAME_ID", "event_num", "xfg"])
merged = pred_df.merge(
    test_df[["GAME_ID", "event_num", "SHOT_MADE_FLAG"]],
    on=["GAME_ID", "event_num"],
    how="left",
)

print("\nPrediction summary (test season):")
print(merged["xfg"].describe())
print("\nAverage xFG by outcome (test season):")
print(merged.groupby("SHOT_MADE_FLAG")["xfg"].mean())

# write to sqlite (disabled by default)
# If you want this, flip WRITE_PRED_TO_SQLITE to True and ensure SQLITE_PATH is valid.
WRITE_PRED_TO_SQLITE = False
if WRITE_PRED_TO_SQLITE:
    con = sqlite3.connect(_resolve_sqlite_path(SQLITE_PATH))
    cur = con.cursor()
    cur.execute(
        """
    CREATE TABLE IF NOT EXISTS shots_xfg_pred (
      GAME_ID TEXT,
      event_num INTEGER,
      xfg REAL,
      PRIMARY KEY (GAME_ID, event_num)
    );
    """
    )
    cur.executemany(
        "INSERT OR REPLACE INTO shots_xfg_pred (GAME_ID, event_num, xfg) VALUES (?, ?, ?);",
        pred_rows,
    )
    con.commit()
    con.close()
    print(f"\nWrote {len(pred_rows)} test predictions into shots_xfg_pred.")
