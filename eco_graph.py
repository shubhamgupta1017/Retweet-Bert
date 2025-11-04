import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === Config ===
EDGE_FILE = "output/edges.csv"
USER_FILE = "data/train.csv"
OUTPUT_IMG = "output/retweet_composition.png"  # path to save the figure

# 👇 Specify users and their display names
TARGET_USERS = {
    "2882331822": "Jenna Armas(Right)",
    "2547141851": "Chicago news(Left)",
    # add more like "userid": "Display Name"
}

# === Load data ===
edges = pd.read_csv(EDGE_FILE)
users = pd.read_csv(USER_FILE)
user_label = dict(zip(users['user'], users['label']))

# === Count retweets by source label ===
counts = []

for target, display_name in TARGET_USERS.items():
    retweets = edges[edges['user1'] == target]
    total = len(retweets)
    left = right = other = 0

    for r in retweets['user2']:
        if r in user_label:
            if user_label[r] == 1:
                right += 1
            else:
                left += 1
        else:
            other += 1

    counts.append({
        'user': display_name,
        'total': total,
        'left': left,
        'right': right,
        'other': other
    })

df = pd.DataFrame(counts)

# === Compute proportions ===
df['left_prop'] = df['left'] / df['total'].replace(0, np.nan)
df['right_prop'] = df['right'] / df['total'].replace(0, np.nan)
df['other_prop'] = df['other'] / df['total'].replace(0, np.nan)

# === Plot ===
fig, ax = plt.subplots(figsize=(8, 0.5 * len(df) + 2))

y = np.arange(len(df))
ax.barh(y, df['left_prop'], color='#1f77b4', label='Left retweeters')
ax.barh(y, df['right_prop'], left=df['left_prop'], color='#d62728', label='Right retweeters')
ax.barh(y, df['other_prop'], left=df['left_prop'] + df['right_prop'], color='lightgray', label='Other')

# Labels
ax.set_yticks(y)
ax.set_yticklabels(df['user'])
ax.invert_yaxis()
ax.set_xlabel('Proportion of retweeters')
ax.set_title('Retweet composition for selected users')

# Total retweet counts on right side
for i, total in enumerate(df['total']):
    ax.text(1.02, i, f"{total:,}", va='center')

ax.legend(loc='lower right')
plt.tight_layout()

# === Save image ===
plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
print(f"✅ Figure saved to {OUTPUT_IMG}")

plt.show()
