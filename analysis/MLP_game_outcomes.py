"""
Game Outcomes v1 — Multi-task MLP (win / margin / total points)

Reads from SQLite table: nba_stats_games
Predicts:
  - home_win (binary)
  - margin (home_score - away_score)
  - total points (home_score + away_score)

Minimal features: team IDs + season/season_type (embeddings).
"""

import os
import sqlite3
import time

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Config
# -----------------------
SQLITE_PATH = "../data/parsed/nba.sqlite"
TABLE = "nba_stats_games"
TEAM_BOX_TABLE = "nba_stats_team_box_traditional"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4096
EPOCHS = 100
LR = 1e-3
SEED = 42
HIDDEN_SIZES = (128, 64)
DROPOUT = 0.2
LOG_EVERY = 50  # batches; set to 0 to disable

EARLY_STOPPING = True
PATIENCE = 25
MONITOR = "win_auc"  # or "margin_rmse" / "total_rmse"

WIN_W = 1.0
MARGIN_W = 0.5
TOTAL_W = 0.5

# Season split (edit if needed)
TRAIN_SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2016, 2024)]  # 2016-17..2023-24
VAL_SEASON = "2024-25"
TEST_SEASON = "2025-26"

ROLL_WINDOWS = (5, 10)
ROLL_STATS = [
    "points",
    "fieldGoalsMade",
    "fieldGoalsAttempted",
    "threePointersMade",
    "threePointersAttempted",
    "freeThrowsMade",
    "freeThrowsAttempted",
    "reboundsOffensive",
    "reboundsDefensive",
    "assists",
    "steals",
    "blocks",
    "turnovers",
]
TEAM_GAME_ROLL_STATS = [
    "points_for",
    "points_against",
    "point_diff",
]

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
    finally:
        con.close()
    return df_


df = load_table(SQLITE_PATH, TABLE)
team_box = load_table(SQLITE_PATH, TEAM_BOX_TABLE)

required_cols = [
    "game_id",
    "season",
    "season_type",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing required columns in {TABLE}: {missing}\n"
        f"Run: PRAGMA table_info({TABLE}); and confirm column names."
    )

df = df.dropna(subset=required_cols).copy()
df["game_id"] = df["game_id"].astype(str)
df["season"] = df["season"].astype(str)
df["season_type"] = df["season_type"].astype(str)
df["home_team_id"] = df["home_team_id"].astype(str)
df["away_team_id"] = df["away_team_id"].astype(str)
df["game_date"] = pd.to_datetime(df["game_date"])

# Targets
df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
df["margin"] = df["home_score"] - df["away_score"]
df["total_points"] = df["home_score"] + df["away_score"]

# Keep only seasons we intend to use
keep_seasons = set(TRAIN_SEASONS + [VAL_SEASON, TEST_SEASON])
df = df[df["season"].isin(keep_seasons)].copy()

if df.empty:
    raise ValueError(
        "After filtering to TRAIN/VAL/TEST seasons, the dataset is empty.\n"
        "Run: SELECT DISTINCT season FROM nba_stats_games ORDER BY season;"
    )

# -----------------------
# Team-level rolling + rest features from game results
# -----------------------
team_games_home = df[
    ["game_id", "game_date", "season", "season_type", "home_team_id", "away_team_id", "home_score", "away_score"]
].rename(
    columns={
        "home_team_id": "team_id",
        "away_team_id": "opp_id",
        "home_score": "points_for",
        "away_score": "points_against",
    }
)
team_games_home["is_home"] = 1

team_games_away = df[
    ["game_id", "game_date", "season", "season_type", "home_team_id", "away_team_id", "home_score", "away_score"]
].rename(
    columns={
        "away_team_id": "team_id",
        "home_team_id": "opp_id",
        "away_score": "points_for",
        "home_score": "points_against",
    }
)
team_games_away["is_home"] = 0

team_games = pd.concat([team_games_home, team_games_away], ignore_index=True)
team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
team_games = team_games.sort_values(["team_id", "season", "season_type", "game_date", "game_id"])

# Rest days: days since last game (per team/season/season_type), shifted to avoid leakage
team_games["rest_days"] = (
    team_games.groupby(["team_id", "season", "season_type"])["game_date"]
    .diff()
    .dt.days
    .fillna(7)
)
team_games["rest_days"] = team_games["rest_days"].clip(lower=0, upper=10)

# Rolling win% and points (shifted)
team_roll_cols = []
for w in ROLL_WINDOWS:
    col = f"win_pct_r{w}"
    team_games[col] = (
        team_games.groupby(["team_id", "season", "season_type"])["win"]
        .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        .fillna(0.5)
    )
    team_roll_cols.append(col)

for stat in TEAM_GAME_ROLL_STATS:
    for w in ROLL_WINDOWS:
        col = f"{stat}_r{w}"
        team_games[col] = (
            team_games.groupby(["team_id", "season", "season_type"])[stat]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        )
        team_roll_cols.append(col)

home_team_feats = team_games[
    ["game_id", "team_id", "rest_days"] + team_roll_cols
].rename(columns={c: f"home_{c}" for c in ["rest_days"] + team_roll_cols})

away_team_feats = team_games[
    ["game_id", "team_id", "rest_days"] + team_roll_cols
].rename(columns={c: f"away_{c}" for c in ["rest_days"] + team_roll_cols})

df = df.merge(
    home_team_feats,
    left_on=["game_id", "home_team_id"],
    right_on=["game_id", "team_id"],
    how="left",
).drop(columns=["team_id"])

df = df.merge(
    away_team_feats,
    left_on=["game_id", "away_team_id"],
    right_on=["game_id", "team_id"],
    how="left",
).drop(columns=["team_id"])

# Diff/sum for team-game rolling
for c in team_roll_cols:
    df[f"diff_{c}"] = df[f"home_{c}"] - df[f"away_{c}"]
    df[f"sum_{c}"] = df[f"home_{c}"] + df[f"away_{c}"]

df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]

# -----------------------
# Rolling team features
# -----------------------
team_box = team_box.copy()
team_box["GAME_ID"] = team_box["GAME_ID"].astype(str)
team_box["team_id"] = team_box["team_id"].astype(str)
team_box["season"] = team_box["season"].astype(str)
team_box["season_type"] = team_box["season_type"].astype(str)

team_box = team_box.merge(
    df[["game_id", "game_date", "season", "season_type"]],
    left_on=["GAME_ID", "season", "season_type"],
    right_on=["game_id", "season", "season_type"],
    how="left",
)

missing_stats = [c for c in ROLL_STATS if c not in team_box.columns]
if missing_stats:
    raise ValueError(
        f"Missing required columns in {TEAM_BOX_TABLE}: {missing_stats}"
    )

team_box = team_box.sort_values(
    ["team_id", "season", "season_type", "game_date", "GAME_ID"]
)

global_means = {c: team_box[c].mean() for c in ROLL_STATS}
roll_cols = []
for stat in ROLL_STATS:
    for w in ROLL_WINDOWS:
        col = f"{stat}_r{w}"
        team_box[col] = (
            team_box.groupby(["team_id", "season", "season_type"])[stat]
            .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
            .fillna(global_means[stat])
        )
        roll_cols.append(col)

home_feat = team_box[["GAME_ID", "team_id"] + roll_cols].rename(
    columns={c: f"home_{c}" for c in roll_cols}
)
away_feat = team_box[["GAME_ID", "team_id"] + roll_cols].rename(
    columns={c: f"away_{c}" for c in roll_cols}
)

df = df.merge(
    home_feat,
    left_on=["game_id", "home_team_id"],
    right_on=["GAME_ID", "team_id"],
    how="left",
).drop(columns=["GAME_ID", "team_id"])

df = df.merge(
    away_feat,
    left_on=["game_id", "away_team_id"],
    right_on=["GAME_ID", "team_id"],
    how="left",
).drop(columns=["GAME_ID", "team_id"])

# Diff and sum features
for stat in ROLL_STATS:
    for w in ROLL_WINDOWS:
        h = f"home_{stat}_r{w}"
        a = f"away_{stat}_r{w}"
        df[f"diff_{stat}_r{w}"] = df[h] - df[a]
        df[f"sum_{stat}_r{w}"] = df[h] + df[a]


# -----------------------
# Feature setup
# -----------------------
cat_cols = ["home_team_id", "away_team_id", "season_type", "season"]
num_cols = []
num_cols.extend(
    ["home_rest_days", "away_rest_days", "diff_rest_days"]
)
for c in team_roll_cols:
    num_cols.extend([f"home_{c}", f"away_{c}", f"diff_{c}", f"sum_{c}"])
for stat in ROLL_STATS:
    for w in ROLL_WINDOWS:
        num_cols.extend(
            [
                f"home_{stat}_r{w}",
                f"away_{stat}_r{w}",
                f"diff_{stat}_r{w}",
                f"sum_{stat}_r{w}",
            ]
        )

# Map categories -> integer ids
cat_maps = {}
cat_cardinalities = {}
for c in cat_cols:
    cats = df[c].unique().tolist()
    cat_maps[c] = {v: i for i, v in enumerate(cats)}
    cat_cardinalities[c] = len(cats)
    df[c + "_id"] = df[c].map(cat_maps[c]).astype(int)

cat_id_cols = [c + "_id" for c in cat_cols]

# Split by season
train_df = df[df["season"].isin(TRAIN_SEASONS)].copy()
val_df = df[df["season"] == VAL_SEASON].copy()
test_df = df[df["season"] == TEST_SEASON].copy()

if train_df.empty or val_df.empty or test_df.empty:
    raise ValueError(
        f"One of your splits is empty.\n"
        f"Train rows: {len(train_df)} | Val rows: {len(val_df)} | Test rows: {len(test_df)}\n"
        "Fix TRAIN_SEASONS / VAL_SEASON / TEST_SEASON to match your data."
    )

if num_cols:
    train_means = train_df[num_cols].mean()
    train_df[num_cols] = train_df[num_cols].fillna(train_means)
    val_df[num_cols] = val_df[num_cols].fillna(train_means)
    test_df[num_cols] = test_df[num_cols].fillna(train_means)

# Scale regression targets using train only
margin_scaler = StandardScaler()
total_scaler = StandardScaler()
train_margin_scaled = margin_scaler.fit_transform(train_df[["margin"]].values)
val_margin_scaled = margin_scaler.transform(val_df[["margin"]].values)
test_margin_scaled = margin_scaler.transform(test_df[["margin"]].values)

train_total_scaled = total_scaler.fit_transform(train_df[["total_points"]].values)
val_total_scaled = total_scaler.transform(val_df[["total_points"]].values)
test_total_scaled = total_scaler.transform(test_df[["total_points"]].values)


# -----------------------
# Dataset / Loader
# -----------------------
class GameDataset(Dataset):
    def __init__(
        self,
        num_x,
        cat_x,
        y_win,
        y_margin_scaled,
        y_total_scaled,
        y_margin,
        y_total,
    ):
        self.num_x = torch.tensor(num_x, dtype=torch.float32)
        self.cat_x = torch.tensor(cat_x, dtype=torch.long)
        self.y_win = torch.tensor(y_win, dtype=torch.float32).view(-1, 1)
        self.y_margin_scaled = torch.tensor(y_margin_scaled, dtype=torch.float32).view(-1, 1)
        self.y_total_scaled = torch.tensor(y_total_scaled, dtype=torch.float32).view(-1, 1)
        self.y_margin = torch.tensor(y_margin, dtype=torch.float32).view(-1, 1)
        self.y_total = torch.tensor(y_total, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.y_win.shape[0]

    def __getitem__(self, idx):
        return (
            self.num_x[idx],
            self.cat_x[idx],
            self.y_win[idx],
            self.y_margin_scaled[idx],
            self.y_total_scaled[idx],
            self.y_margin[idx],
            self.y_total[idx],
        )


def _num_array(d: pd.DataFrame, cols):
    if not cols:
        return np.zeros((len(d), 0), dtype=np.float32)
    return d[cols].values.astype(np.float32)


if num_cols:
    num_scaler = StandardScaler()
    train_num = num_scaler.fit_transform(train_df[num_cols].values)
    val_num = num_scaler.transform(val_df[num_cols].values)
    test_num = num_scaler.transform(test_df[num_cols].values)
else:
    train_num = _num_array(train_df, num_cols)
    val_num = _num_array(val_df, num_cols)
    test_num = _num_array(test_df, num_cols)

train_ds = GameDataset(
    train_num,
    train_df[cat_id_cols].values,
    train_df["home_win"].values,
    train_margin_scaled,
    train_total_scaled,
    train_df["margin"].values,
    train_df["total_points"].values,
)
val_ds = GameDataset(
    val_num,
    val_df[cat_id_cols].values,
    val_df["home_win"].values,
    val_margin_scaled,
    val_total_scaled,
    val_df["margin"].values,
    val_df["total_points"].values,
)
test_ds = GameDataset(
    test_num,
    test_df[cat_id_cols].values,
    test_df["home_win"].values,
    test_margin_scaled,
    test_total_scaled,
    test_df["margin"].values,
    test_df["total_points"].values,
)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# -----------------------
# Model
# -----------------------
def emb_dim(n):
    return min(50, (n + 1) // 2)


class GameMLP(nn.Module):
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
        self.backbone = nn.Sequential(*layers)
        self.win_head = nn.Linear(in_dim, 1)
        self.margin_head = nn.Linear(in_dim, 1)
        self.total_head = nn.Linear(in_dim, 1)

    def forward(self, x_num, x_cat):
        embs = []
        for i, e in enumerate(self.embs):
            embs.append(e(x_cat[:, i]))
        x = torch.cat([x_num] + embs, dim=1)
        h = self.backbone(x)
        return self.win_head(h), self.margin_head(h), self.total_head(h)


model = GameMLP(cat_cardinalities, n_num=len(num_cols), hidden=HIDDEN_SIZES, dropout=DROPOUT).to(
    DEVICE
)
opt = torch.optim.AdamW(model.parameters(), lr=LR)
crit_win = nn.BCEWithLogitsLoss()
crit_reg = nn.MSELoss()

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
    all_win, all_p = [], []
    all_margin_true, all_margin_pred = [], []
    all_total_true, all_total_pred = [], []

    for batch in dl:
        x_num, x_cat, y_win, y_margin_s, y_total_s, y_margin, y_total = batch
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        win_logit, margin_s, total_s = model(x_num, x_cat)

        p = torch.sigmoid(win_logit).cpu().numpy().reshape(-1)
        all_p.append(p)
        all_win.append(y_win.numpy().reshape(-1))

        margin_pred = margin_scaler.inverse_transform(
            margin_s.cpu().numpy().reshape(-1, 1)
        ).reshape(-1)
        total_pred = total_scaler.inverse_transform(
            total_s.cpu().numpy().reshape(-1, 1)
        ).reshape(-1)

        all_margin_pred.append(margin_pred)
        all_total_pred.append(total_pred)
        all_margin_true.append(y_margin.numpy().reshape(-1))
        all_total_true.append(y_total.numpy().reshape(-1))

    y_true = np.concatenate(all_win)
    y_prob = np.concatenate(all_p)

    margin_true = np.concatenate(all_margin_true)
    margin_pred = np.concatenate(all_margin_pred)
    total_true = np.concatenate(all_total_true)
    total_pred = np.concatenate(all_total_pred)

    out = {
        "win_acc": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
        "win_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "margin_mae": float(np.mean(np.abs(margin_true - margin_pred))),
        "margin_rmse": float(np.sqrt(np.mean((margin_true - margin_pred) ** 2))),
        "total_mae": float(np.mean(np.abs(total_true - total_pred))),
        "total_rmse": float(np.sqrt(np.mean((total_true - total_pred) ** 2))),
    }
    return out


for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    running = 0.0
    n = 0
    for batch_idx, batch in enumerate(train_dl, start=1):
        x_num, x_cat, y_win, y_margin_s, y_total_s, _, _ = batch
        x_num, x_cat = x_num.to(DEVICE), x_cat.to(DEVICE)
        y_win = y_win.to(DEVICE)
        y_margin_s = y_margin_s.to(DEVICE)
        y_total_s = y_total_s.to(DEVICE)

        opt.zero_grad(set_to_none=True)
        win_logit, margin_s, total_s = model(x_num, x_cat)

        loss = (
            WIN_W * crit_win(win_logit, y_win)
            + MARGIN_W * crit_reg(margin_s, y_margin_s)
            + TOTAL_W * crit_reg(total_s, y_total_s)
        )
        loss.backward()
        opt.step()
        running += loss.item() * y_win.shape[0]
        n += y_win.shape[0]

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
        f"val_win_acc={val_metrics['win_acc']:.4f} | val_win_auc={val_metrics['win_auc']:.4f} | "
        f"val_margin_rmse={val_metrics['margin_rmse']:.2f} | val_total_rmse={val_metrics['total_rmse']:.2f} | "
        f"time={dt:.1f}s"
    )

    # Early stopping on selected metric
    if epoch == 1:
        best_metric = val_metrics[MONITOR]
        best_epoch = 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        improved = (
            val_metrics[MONITOR] > best_metric
            if MONITOR == "win_auc"
            else val_metrics[MONITOR] < best_metric
        )
        if improved:
            best_metric = val_metrics[MONITOR]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if EARLY_STOPPING and no_improve >= PATIENCE:
                print(
                    f"Early stopping at epoch {epoch} (best {MONITOR}={best_metric:.4f} at epoch {best_epoch})."
                )
                break

if EARLY_STOPPING:
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    print(f"Loaded best model from epoch {best_epoch} (best {MONITOR}={best_metric:.4f}).")

test_metrics = eval_loader(test_dl)
print("TEST:", test_metrics)
