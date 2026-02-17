import os
import pandas as pd
import matplotlib.pyplot as plt

# results.csv 경로 (스크립트와 같은 폴더)
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "results_last.csv")

print("CSV path:", csv_path)
df = pd.read_csv(csv_path)

epoch_col = "epoch" if "epoch" in df.columns else df.columns[0]

panel_cols = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
]

rows, cols = 2, 5
fig, axes = plt.subplots(rows, cols, figsize=(18, 8), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(panel_cols):
    ax = axes[i]
    if col not in df.columns:
        ax.set_axis_off()
        continue

    ax.plot(df[epoch_col], df[col], marker="o", markersize=3)
    ax.set_title(col)
    ax.grid(True, alpha=0.3)

# 혹시 컬럼 부족하면 남는 칸 숨김
for j in range(len(panel_cols), len(axes)):
    axes[j].set_axis_off()

out_path = os.path.join(base_dir, "results_last.png")
plt.savefig(out_path, dpi=300)
plt.show()

print("Saved as:", out_path)
