import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib import rcParams
import numpy as np

# ---------- Style ----------
rcParams.update({
    "font.family": 'serif',
    "font.size": 11,
    "mathtext.fontset": 'stix',
    "font.serif": ['Times New Roman'],
})

def norm(v):
    return v * 100 if v <= 1 else v


def load_eval_series(base_dir, metric_key, block_key):
    series = {}
    for model_dir in sorted(path for path in base_dir.iterdir() if path.is_dir()):
        history_path = model_dir / "eval_history.jsonl"
        if not history_path.exists():
            continue
        steps = []
        values = []
        with history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record.get("split") != "dev":
                    continue
                step = record.get("step")
                if step is None:
                    continue
                block = record.get(block_key, {})
                if metric_key in block:
                    steps.append(step)
                    values.append(block[metric_key])
        if steps:
            series[model_dir.name] = (steps, values)
    return series


def sorted_series(series):
    steps, values = series
    pairs = sorted(zip(steps, values), key=lambda item: item[0])
    sorted_steps = [step for step, _ in pairs]
    sorted_values = [value for _, value in pairs]
    return sorted_steps, sorted_values

base_dir = Path(__file__).resolve().parent
series_by_model = load_eval_series(base_dir, "success@5", "representation")
if not series_by_model:
    raise SystemExit("No success@5 series found in eval_history.jsonl files.")

max_step = 3000
labels = list(range(0, max_step + 1, 500))
x = list(range(len(labels)))

names = []
series = []
for model_name in sorted(series_by_model.keys()):
    steps, values = sorted_series(series_by_model[model_name])
    values = list(map(norm, values))
    if len(steps) == 1:
        interp_values = [values[0]] * len(labels)
    else:
        interp_values = np.interp(labels, steps, values)
    names.append(model_name)
    series.append(interp_values)

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_facecolor("#eef7f1")

all_vals = [v for arr in series for v in arr]
base_y = min(all_vals) - 3
top_y  = max(all_vals) + 3

colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3'])

for (y, name, c) in zip(series, names, colors):
    # 阴影
    ax.fill_between(x, y, base_y, facecolor=c, alpha=0.15, zorder=1)
    # 线条 + 发光效果
    (ln,) = ax.plot(x, y, color=c, linewidth=3.8, label=name, zorder=3)
    ln.set_path_effects([
        pe.Stroke(linewidth=8, foreground=c, alpha=0.35),
        pe.Normal(),
    ])

# 轴网格与边框
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_linewidth(2.0)
ax.spines["bottom"].set_linewidth(2.0)

# ---------- 设置 X 轴 ----------
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)

# 设置显示范围，负值越小，0点离Y轴越远；这里设为 -0.1 比较紧凑
ax.set_xlim(-0.1, len(x)-0.9)
ax.set_ylim(base_y, top_y)

ax.tick_params(axis='x', labelsize=27)
ax.tick_params(axis='y', labelsize=27)
ax.set_ylabel("Success@5", fontsize=32)
ax.set_xlabel("Training Steps", fontsize=32)

ax.legend(loc="lower right", ncol=2, framealpha=0.9, fontsize=20)

fig.tight_layout()
output_dir = base_dir / "plots"
output_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(output_dir / "success5.png", dpi=300, bbox_inches="tight")
fig.savefig(output_dir / "success5.pdf", bbox_inches="tight")
plt.close(fig)
