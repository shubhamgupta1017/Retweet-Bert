import pandas as pd
from textblob import TextBlob

# === CONFIG ===
INPUT_FILE = "data/ira_tweets_csv_hashed.csv"   # your raw tweets CSV
USERS_OUT = "output/users_bin.csv"             # output file for user profiles
EDGES_OUT = "output/edges_bin.csv"             # output file for sentiment-weighted retweet edges
CHUNKSIZE = 100_000                             # adjust if you have more/less RAM

print("🚀 Processing large CSV (English tweets only, sentiment-based weights)...\n")

user_dict = {}
edge_list = []

def sentiment_to_edge_weight(polarity):
    """Convert polarity to +1, -1, or 0."""
    if polarity > 0:
        return 1
    elif polarity < 0:
        return -1
    else:
        return 0

for i, chunk in enumerate(pd.read_csv(INPUT_FILE, chunksize=CHUNKSIZE)):

    print(f"🔹 Processing chunk {i + 1}...")

    chunk.columns = [c.strip().lower() for c in chunk.columns]

    # --- keep only English tweets ---
    if "tweet_language" in chunk.columns:
        chunk = chunk[chunk["tweet_language"] == "en"]

    # --- users (id + description) ---
    if "userid" in chunk.columns and "user_profile_description" in chunk.columns:
        users_chunk = chunk[["userid", "user_profile_description"]].dropna()
        for _, row in users_chunk.iterrows():
            uid = str(row["userid"])
            if uid not in user_dict:
                user_dict[uid] = str(row["user_profile_description"])

    # --- sentiment-weighted retweets ---
    if (
        "is_retweet" in chunk.columns
        and "retweet_userid" in chunk.columns
        and "userid" in chunk.columns
        and "tweet_text" in chunk.columns
    ):
        rt_chunk = chunk[chunk["is_retweet"] == True].dropna(subset=["tweet_text"])

        # convert sentiment polarity to +1, -1, 0
        rt_chunk["sentiment_weight"] = rt_chunk["tweet_text"].apply(
            lambda text: sentiment_to_edge_weight(TextBlob(str(text)).sentiment.polarity)
        )

        # remove zero-weight edges
        rt_chunk = rt_chunk[rt_chunk["sentiment_weight"] != 0]

        rt_edges = rt_chunk[["retweet_userid", "userid", "sentiment_weight"]]
        edge_list.append(rt_edges)

    print(f"   ✅ Users so far: {len(user_dict):,}")

# === Combine all chunks ===
print("\n🧩 Combining all edges...")
if edge_list:
    edges_df = pd.concat(edge_list, ignore_index=True)
    edges_df.columns = ["user1", "user2", "weight"]

    print("📊 Averaging sentiment weights per edge...")
    # average sentiment per unique pair and round to nearest +1/-1
    edges_df = (
        edges_df.groupby(["user1", "user2"])["weight"]
        .mean()
        .round()
        .astype(int)
        .reset_index()
        .sort_values(by="weight", ascending=False)
    )

    # remove any 0-weight edges after averaging
    edges_df = edges_df[edges_df["weight"] != 0]

    # keep only users that exist in user_dict
    valid_users = set(user_dict.keys())
    edges_df = edges_df[
        edges_df["user1"].isin(valid_users) & edges_df["user2"].isin(valid_users)
    ].reset_index(drop=True)
else:
    print("⚠️ No retweet edges found!")
    edges_df = pd.DataFrame(columns=["user1", "user2", "weight"])

# === Save outputs ===
print("💾 Writing output files...")
pd.DataFrame(list(user_dict.items()), columns=["user", "profile"]).to_csv(
    USERS_OUT, index=False
)
edges_df.to_csv(EDGES_OUT, index=False)

print(f"\n✅ Done! Users: {len(user_dict):,}, Edges: {len(edges_df):,}")
print("   (Edge weights = +1 or -1 based on average sentiment polarity)")
