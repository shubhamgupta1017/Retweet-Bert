
import pandas as pd

INPUT_FILE = "data/ira_tweets_csv_hashed.csv"       # your raw tweets CSV
USERS_OUT = "output/users.csv"         # output file for user profiles
EDGES_OUT = "output/edges.csv"         # output file for retweet edges
CHUNKSIZE = 100_000  # adjust if you have more/less RAM

print("🚀 Processing large CSV (English tweets only)...\n")

user_dict = {}
edge_list = []

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

    # --- edges (retweet relations) ---
    if "is_retweet" in chunk.columns:
        rt_chunk = chunk[chunk["is_retweet"] == True]
        if "retweet_userid" in rt_chunk.columns and "userid" in rt_chunk.columns:
            rt_edges = rt_chunk[["retweet_userid", "userid"]].dropna()
            edge_list.append(rt_edges)

    print(f"   ✅ Users so far: {len(user_dict):,}")

print("\n🧩 Combining all edges...")
if edge_list:
    edges_df = pd.concat(edge_list, ignore_index=True)
    edges_df.columns = ["user1", "user2"]

    print("📊 Computing edge weights...")
    edges_df = edges_df.groupby(["user1", "user2"]).size().reset_index(name="weight")

    valid_users = set(user_dict.keys())
    edges_df = edges_df[
        edges_df["user1"].isin(valid_users) & edges_df["user2"].isin(valid_users)
    ].reset_index(drop=True)
else:
    print("⚠️ No edges found in English tweets!")
    edges_df = pd.DataFrame(columns=["user1", "user2", "weight"])

# --- save ---
print("💾 Writing output files...")
pd.DataFrame(list(user_dict.items()), columns=["user", "profile"]).to_csv(
    USERS_OUT, index=False
)
edges_df.to_csv(EDGES_OUT, index=False)

print(f"\n✅ Done! Users: {len(user_dict):,}, Edges: {len(edges_df):,}")
