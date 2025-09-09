import os
import re
import ast
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ================= Optional deps (DL mode, etc.) =================
HAS_ST = False
HAS_SK = False
HAS_SM = False
HAS_TRANS = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.metrics import silhouette_score  # type: ignore
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    HAS_SK = True
except Exception:
    HAS_SK = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    HAS_SM = True
except Exception:
    HAS_SM = False

# NEW: translation helpers (best-effort; safe fallback to original text)
try:
    from langdetect import detect  # type: ignore
    HAS_TRANS = True
except Exception:
    HAS_TRANS = False

try:
    # googletrans can fail without internet; we guard with try/except when calling
    from googletrans import Translator  # type: ignore
    translator = Translator()
except Exception:
    translator = None

st.set_page_config(page_title="TrendSpotter", layout="wide")

# ---------------- Utilities ----------------
def safe_str(x):
    return "" if pd.isna(x) else str(x)

hashtag_pattern = re.compile(r"#(\w+)")
word_pattern = re.compile(r"[A-Za-z][A-Za-z0-9_]+")
stopwords = set("""
a an and the or for of to in on with by from at is are was were be been being
this that those these it its as if but so not no you your our their my me we us
i they he she them him her than then there here when where how what which who
lol lmao rofl omg tbh idk bruh bro sis slay slaying period fr frfr cap nocap no-cap
haha hehe huhu yass yasss ayy ayo aiyo oop oops missed sunday known regular won jesus 
ridiculous died naa fly senior
""".split())

def parse_tags(val: str):
    val = val.strip()
    tags = []
    if not val:
        return tags
    if val.startswith("[") and val.endswith("]"):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list):
                tags = [str(t).strip() for t in parsed if str(t).strip()]
        except Exception:
            pass
    if not tags:
        sep = "|" if "|" in val else ","
        rough = [t.strip() for t in val.split(sep)]
        tags = [t.strip(" '\"[]") for t in rough if t.strip(" '\"[]")]
    return [t for t in tags if t]

def unify_token(t: str) -> str:
    unify_map = {
        r"^sonakshi\s*sinha$|^sonakshisinha$|^sonakshi$|^sinha$": "sonakshi_sinha",
        r"^cosmetic\s*crafts$|^cosmeticcrafts$": "cosmetic_crafts",
        r"^haircut\s*power$": "haircut_power",
        r"^hair\s*before\s*after$": "hair_before_after",
    }
    s = str(t)
    for pattern, repl in unify_map.items():
        if re.match(pattern, s):
            return repl
    return re.sub(r"\s+", "_", s)

# NEW: always-on translation (titles/descriptions). Best-effort with safe fallback.
def translate_text_auto(s: str) -> str:
    s = safe_str(s)
    if not s or not HAS_TRANS:
        return s
    try:
        lang = detect(s)
        if lang and lang.lower().startswith("en"):
            return s  # already English
        if translator is None:
            return s  # no translator available; keep original
        # translate to English
        t = translator.translate(s, dest="en")
        return t.text if t and t.text else s
    except Exception:
        return s

# NEW: Q9—emoji/encoding cleanup for comments (only if comments are uploaded)
def clean_comment_text(s: str) -> str:
    import unicodedata
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    s = re.sub(r"[\x00-\x1F\x7F-\x9F\u200B-\u200D\uFEFF]", " ", s)
    s = s.replace("\r", " ").replace("\n", " ").strip()
    return s

def extract_tokens(title, description, tags_list):
    tokens = []
    hashtags = hashtag_pattern.findall(title + " " + description)
    tokens.extend([h.lower() for h in hashtags])
    tokens.extend([t.lower() for t in tags_list if len(t) >= 2])
    words = [w.lower() for w in word_pattern.findall(title)]
    words = [w for w in words if len(w) >= 3 and w not in stopwords]
    tokens.extend(words)
    return list(set(tokens))

def map_segment(topic_str):
    s = str(topic_str).lower()
    if "beauty" in s or "cosmetics" in s or "makeup" in s or "skin_care" in s or "hairstyle" in s:
        return "Beauty"
    if "lifestyle" in s or "fashion" in s or "food" in s:
        return "Lifestyle"
    if "sports" in s or "game" in s or "athletics" in s:
        return "Sports"
    if "health" in s or "fitness" in s or "medicine" in s:
        return "Health"
    return "Other"

@st.cache_data(show_spinner=False)
def load_df(path_or_bytes) -> pd.DataFrame:
    if isinstance(path_or_bytes, (str, os.PathLike)) and os.path.exists(path_or_bytes):
        df = pd.read_csv(path_or_bytes)
    else:
        df = pd.read_csv(path_or_bytes)

    expected_cols = {
        "videoId","publishedAt","title","description","tags","topicCategories",
        "viewCount","likeCount","commentCount","defaultLanguage","defaultAudioLanguage","contentDuration"
    }
    for c in expected_cols:
        if c not in df.columns:
            df[c] = np.nan

    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
    df = df.dropna(subset=["publishedAt"]).copy()

    # Convert to MYT; keep naive copy for masks
    df["publishedAt_local"] = df["publishedAt"].dt.tz_convert("Asia/Kuala_Lumpur")
    df["published_local_naive"] = df["publishedAt_local"].dt.tz_localize(None)

    # Normalize text & translate (NEW)
    for col in ["title","description","tags","topicCategories"]:
        df[col] = df[col].apply(safe_str)

    # Always-on translation for titles/descriptions (best-effort)
    df["title"] = df["title"].apply(translate_text_auto)
    df["description"] = df["description"].apply(translate_text_auto)

    # Ensure numeric engagement columns
    for col in ["viewCount","likeCount","commentCount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Parse tags, build tokens
    df["tags_list"] = df["tags"].apply(parse_tags)
    df["tokens"] = df.apply(lambda r: extract_tokens(r["title"], r["description"], r["tags_list"]), axis=1)
    df = df[df["tokens"].map(len) > 0].copy()
    df["week"] = df["publishedAt_local"].dt.tz_localize(None).dt.to_period("W").apply(lambda p: p.start_time)

    # NEW: derive audio_lang proxy for “audio” analysis
    df["defaultLanguage"] = df["defaultLanguage"].fillna("und").astype(str).str.lower()
    df["defaultAudioLanguage"] = df["defaultAudioLanguage"].fillna("und").astype(str).str.lower()
    df["audio_lang"] = np.where(df["defaultAudioLanguage"].ne("und"), df["defaultAudioLanguage"], df["defaultLanguage"])

    return df

def compute_windows(df: pd.DataFrame, recent_days: int, prev_days: int):
    end_date     = df["published_local_naive"].max().normalize() + pd.Timedelta(days=1)
    start_recent = end_date - pd.Timedelta(days=recent_days)
    start_prev   = start_recent - pd.Timedelta(days=prev_days)
    return start_prev, start_recent, end_date

def compute_trends(df: pd.DataFrame, recent_days: int, prev_days: int, min_recent: int):
    start_prev, start_recent, end_date = compute_windows(df, recent_days, prev_days)
    exploded = df[["title","videoId","publishedAt_local","published_local_naive","week","tokens","topicCategories","viewCount","likeCount","commentCount"]] \
                .explode("tokens").rename(columns={"tokens":"token"})
    exploded["token"] = exploded["token"].astype(str).str.replace("^#", "", regex=True).apply(unify_token)
    exploded = exploded[exploded["token"].str.len() >= 3]
    weekly_counts = exploded.groupby(["week","token"]).size().reset_index(name="count").sort_values(["token","week"])

    def window_sum(df_counts, start, end):
        m = (df_counts["week"] >= start) & (df_counts["week"] < end)
        return df_counts[m].groupby("token")["count"].sum().rename("count").reset_index()

    recent = window_sum(weekly_counts, start_recent, end_date)
    prev = window_sum(weekly_counts, start_prev, start_recent)

    trend = recent.merge(prev, on="token", how="left", suffixes=("_recent","_prev")).fillna({"count_prev":0})
    trend["growth_abs"] = trend["count_recent"] - trend["count_prev"]
    trend["growth_rate"] = trend.apply(lambda r: (r["count_recent"]/r["count_prev"]) if r["count_prev"]>0 else (np.inf if r["count_recent"]>0 else 0), axis=1)
    trend = trend[trend["count_recent"] >= min_recent].copy()
    trend["rank_score"] = trend["growth_abs"] * np.where(np.isinf(trend["growth_rate"]), 3, np.minimum(trend["growth_rate"], 5))

    eng = exploded.groupby("token")[["viewCount","likeCount","commentCount"]].sum().fillna(0).reset_index()
    trend_m = trend.merge(eng, on="token", how="left").fillna(0)
    trend_m["momentum_score"] = trend_m["rank_score"] * (trend_m["likeCount"] + 1)

    return trend, trend_m, weekly_counts, exploded, (start_prev, start_recent, end_date)

def dl_topics(df: pd.DataFrame, recent_days: int, prev_days: int):
    start_prev, start_recent, end_date = compute_windows(df, recent_days, prev_days)
    mask = (df["published_local_naive"] >= start_recent) & (df["published_local_naive"] < end_date)
    titles_recent = df.loc[mask, "title"].dropna().astype(str)

    if len(titles_recent) < 50 or not HAS_SK:
        return None, None

    try:
        if HAS_ST:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            titles_sample = titles_recent.sample(min(5000, len(titles_recent)), random_state=42)
            X = model.encode(titles_sample.tolist(), normalize_embeddings=True)

            best_k, best_score, best_km = None, -1, None
            for k in [6,8,10,12]:
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels_try = km.fit_predict(X)
                try:
                    score = silhouette_score(X, labels_try)
                except Exception:
                    score = -1
                if score > best_score:
                    best_k, best_score, best_km = k, score, km

            labels = best_km.predict(X)

            import collections
            def words(s): return re.findall(r"[A-Za-z][A-Za-z0-9_]+", s.lower())
            stop = set("""a an the and or for of to in on with by from this that those these it its as if but so not no you your our their my me we us i they he she them him her than then there here when where how what which who is are was were be been being""".split())

            cluster_titles = pd.DataFrame({"title": titles_sample.values, "cluster": labels})
            topic_labels = {}
            for c, grp in cluster_titles.groupby("cluster"):
                from collections import Counter
                cnt = Counter(w for t in grp["title"] for w in words(t) if len(w)>=3 and w not in stop)
                common = [w for w,_ in cnt.most_common(5)]
                topic_labels[c] = ", ".join(common[:3]) if common else f"topic_{c}"

            df_recent = df.loc[mask, ["title","published_local_naive"]].dropna().copy()
            Z = model.encode(df_recent["title"].tolist(), normalize_embeddings=True)
            cluster_ids = best_km.predict(Z)
            df_recent["topic_label"] = [topic_labels.get(c, f"topic_{c}") for c in cluster_ids]
        else:
            raise RuntimeError("Transformer not available")
    except Exception:
        # Fallback TF-IDF
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1,2), min_df=3)
        X = vec.fit_transform(titles_recent)
        km = KMeans(n_clusters=10, n_init=10, random_state=42).fit(X)
        df_recent = pd.DataFrame({
            "title": titles_recent.values,
            "published_local_naive": df.loc[mask, "published_local_naive"].values[:len(titles_recent)]
        })
        df_recent["topic_label"] = km.labels_.astype(str)

    df_recent["week"] = df_recent["published_local_naive"].dt.to_period("W").apply(lambda p: p.start_time)
    topic_weekly = df_recent.groupby(["week","topic_label"]).size().reset_index(name="count").sort_values(["topic_label","week"])

    def window_sum_topic(dfc, start, end):
        m = (dfc["week"] >= start) & (dfc["week"] < end)
        return dfc[m].groupby("topic_label")["count"].sum().rename("count").reset_index()

    topic_recent = window_sum_topic(topic_weekly, start_recent, end_date)
    topic_prev   = window_sum_topic(topic_weekly, start_prev, start_recent)
    topic_trend = topic_recent.merge(topic_prev, on="topic_label", how="left", suffixes=("_recent","_prev")).fillna({"count_prev":0})
    topic_trend["growth_abs"]  = topic_trend["count_recent"] - topic_trend["count_prev"]
    topic_trend["growth_rate"] = topic_trend.apply(lambda r: (r["count_recent"]/r["count_prev"]) if r["count_prev"]>0 else (np.inf if r["count_recent"]>0 else 0), axis=1)
    topic_trend = topic_trend[topic_trend["count_recent"] >= 5].copy()
    topic_trend["rank_score"] = topic_trend["growth_abs"] * np.where(np.isinf(topic_trend["growth_rate"]), 3, np.minimum(topic_trend["growth_rate"], 5))
    top_topics = topic_trend.sort_values(["rank_score","count_recent"], ascending=[False, False]).head(10)
    return top_topics, df_recent

def ensure_exploded_with_title(df_base: pd.DataFrame) -> pd.DataFrame:
    df2 = df_base.copy()
    if "week" not in df2.columns:
        df2["week"] = df2["publishedAt_local"].dt.tz_localize(None).dt.to_period("W").apply(lambda p: p.start_time)
    exploded_ok = (
        df2[["title","videoId","publishedAt_local","week","tokens","topicCategories",
             "viewCount","likeCount","commentCount","audio_lang"]]
        .explode("tokens")
        .rename(columns={"tokens":"token"})
        .dropna(subset=["title","token"])
        .copy()
    )
    exploded_ok["token"] = (
        exploded_ok["token"].astype(str)
        .str.replace("^#", "", regex=True)
        .apply(unify_token)
    )
    exploded_ok = exploded_ok[exploded_ok["token"].str.len() >= 3]
    return exploded_ok

def build_hybrid_simple(trend, trend_m, top_topics, df_recent, exploded, alpha=0.6, beta=0.2):
    if df_recent is None or df_recent.empty or top_topics is None or top_topics.empty:
        return None
    titles_recent = df_recent["title"].unique().tolist()
    recent_exploded = exploded[exploded["title"].isin(titles_recent)][["title","token"]].dropna().drop_duplicates()
    tok_topic = (recent_exploded.merge(df_recent[["title","topic_label"]], on="title", how="inner")
                 .groupby(["token","topic_label"]).size().reset_index(name="n")
                 .sort_values(["token","n"], ascending=[True, False])
                 .drop_duplicates("token")
                 .rename(columns={"topic_label":"mapped_topic", "n":"cooccurrences"}))

    token_scores = trend[["token","rank_score"]].rename(columns={"rank_score":"token_score"}).copy()
    mom = trend_m[["token","momentum_score"]].rename(columns={"momentum_score":"mom_score"}).copy()
    topic_scores = top_topics[["topic_label","rank_score"]].rename(columns={"topic_label":"mapped_topic","rank_score":"topic_score"}).copy()

    topic_weight = max(0.0, 1.0 - alpha - beta)
    hybrid = (token_scores
              .merge(tok_topic[["token","mapped_topic"]], on="token", how="left")
              .merge(topic_scores, on="mapped_topic", how="left")
              .merge(mom, on="token", how="left")
              .fillna({"topic_score":0.0, "mom_score":0.0}))

    hybrid["hybrid_score"] = alpha*hybrid["token_score"] + topic_weight*hybrid["topic_score"] + beta*hybrid["mom_score"]
    return hybrid.sort_values("hybrid_score", ascending=False)

# NEW: Audio-language — all-time distribution + token×language crosstab
def audio_language_views(exploded, top_tokens):
    aud_total = (exploded.groupby("audio_lang")["videoId"].nunique()
                 .reset_index(name="videos").sort_values("videos", ascending=False))
    cross = (exploded[exploded["token"].isin(top_tokens)]
             .groupby(["token","audio_lang"]).size().reset_index(name="videos"))
    matrix = cross.pivot(index="token", columns="audio_lang", values="videos").fillna(0).astype(int)
    matrix_pct = (matrix.div(matrix.sum(axis=1), axis=0).replace([np.inf, np.nan], 0.0).round(3))
    return aud_total, cross, matrix, matrix_pct

# NEW: Evaluation (Beauty Precision@10 + Lead-time)
BEAUTY_TERMS = set('hair haircut hairstyle ponytail dye scalp hairfall skin skincare glow matte lipstick mascara foundation toner serum cleanser makeup fragrance perfume eau nail mani pedi brow lash contour blush highlight retinol hyaluronic'.split())
def is_beauty_token(tok: str):
    t = str(tok).lower()
    return any(w in t for w in BEAUTY_TERMS)

def precision_at_k(tokens, k=10):
    if not tokens:
        return np.nan
    topk = tokens[:k]
    rel = sum(is_beauty_token(t) for t in topk)
    return rel / len(topk)

def lead_time_weeks(weekly_counts, token, lookback_months=12):
    series = (weekly_counts[weekly_counts["token"]==token]
              .sort_values("week").set_index("week")["count"])
    if series.empty:
        return np.nan, np.nan
    raw_weeks = (series.idxmax() - series.index.min()).days/7
    # last N months window
    end = series.index.max()
    start = end - pd.Timedelta(days=30*lookback_months)
    s2 = series[(series.index>=start) & (series.index<=end)]
    adj_weeks = (s2.idxmax() - s2.index.min()).days/7 if len(s2)>0 else np.nan
    return raw_weeks, adj_weeks

# ---------------- Sidebar ----------------
st.sidebar.title("TrendSpotter Controls")
st.sidebar.caption("Dataset & Parameters")

uploaded = st.sidebar.file_uploader("Upload videos.csv", type=["csv"])
default_path = "videos.csv" if os.path.exists("videos.csv") else None
data_source = uploaded if uploaded is not None else (default_path if default_path else None)

# Optional comments enrichment (drop any CSV named comments*.csv)
comments_files = st.sidebar.file_uploader("Upload comments CSV (optional, can multi-select)", type=["csv"], accept_multiple_files=True)

recent_days = st.sidebar.number_input("Recent window (days)", 7, 120, 30, 1)
prev_days   = st.sidebar.number_input("Previous window (days)", 7, 120, 30, 1)
min_recent  = st.sidebar.number_input("Min recent mentions per token", 1, 50, 5, 1)

alpha = st.sidebar.slider("Hybrid: token weight (α)", 0.0, 1.0, 0.6, 0.05)
beta  = st.sidebar.slider("Hybrid: momentum weight (β)", 0.0, 1.0, 0.2, 0.05)
st.sidebar.caption("Topic weight = 1 − α − β")

st.sidebar.markdown("---")
st.sidebar.write(f"SentenceTransformer available: **{HAS_ST}**")
st.sidebar.write(f"Scikit-learn available: **{HAS_SK}**")
st.sidebar.write(f"Statsmodels available: **{HAS_SM}**")
st.sidebar.write(f"Translation available: **{HAS_TRANS and translator is not None}**")

# ---------------- Main ----------------
st.title("L'Oréal × Monash — TrendSpotter")
st.write("**Early trend identification** across hashtags/keywords + **semantic topics**. Identify segments and **when trends decay**. Auto-translates non-English text before tokenization.")

if data_source is None:
    st.warning("Please upload `videos.csv` (or place it in the same folder).")
    st.stop()

with st.spinner("Loading & processing dataset..."):
    df = load_df(data_source)

    # OPTIONAL: merge comments enrichment (Q9 cleaning) — if user uploaded any
    if comments_files:
        frames = []
        needed = {"commentId","videoId","textOriginal","likeCount","publishedAt"}
        for f in comments_files:
            try:
                cdf = pd.read_csv(f)
                for c in needed:
                    if c not in cdf.columns:
                        cdf[c] = np.nan
                cdf = cdf[list(needed)].copy()
                cdf["textOriginal"] = cdf["textOriginal"].fillna("").astype(str).apply(clean_comment_text)
                cdf["videoId"] = cdf["videoId"].astype(str)
                frames.append(cdf)
            except Exception:
                pass
        if frames:
            com_all = pd.concat(frames, ignore_index=True)
            if "commentId" in com_all.columns:
                com_all = com_all.drop_duplicates(subset=["commentId"])
            com_all = com_all.drop_duplicates()
            # aggregate brief comment blob per video (kept short to avoid noise)
            com_agg = (com_all.groupby("videoId")["textOriginal"]
                       .apply(lambda s: " ".join(s.tolist())[:20000]).reset_index()
                       .rename(columns={"textOriginal":"comments_text"}))
            df["videoId"] = df["videoId"].astype(str)
            df = df.merge(com_agg, on="videoId", how="left")
            df["comments_text"] = df["comments_text"].fillna("")
            # enrich tokens lightly with hashtags & words from comments
            def extract_tokens_with_comments(row):
                title, desc, tags_list, comments = row["title"], row["description"], row["tags_list"], row["comments_text"]
                hashtags = hashtag_pattern.findall(f"{title} {desc} {comments}")
                toks = [h.lower() for h in hashtags]
                toks.extend([t.lower() for t in tags_list if len(t) >= 2])
                words = [w.lower() for w in word_pattern.findall(title + " " + comments)]
                words = [w for w in words if len(w) >= 3 and w not in stopwords]
                toks.extend(words)
                return list(set(toks))
            df["tokens"] = df.apply(extract_tokens_with_comments, axis=1)
            df = df[df["tokens"].map(len) > 0].copy()

    trend, trend_m, weekly_counts, exploded_raw, windows = compute_trends(df, recent_days, prev_days, min_recent)
    start_prev, start_recent, end_date = windows

# Tabs (ADDED: Audio-Language, Evaluation)
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Trends", "Momentum", "Segments", "Decay", "Semantic Topics (DL)", "Hybrid", "Audio-Language", "Evaluation"
])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Videos", f"{len(df):,}")
    c2.metric("Unique Tokens", f"{trend['token'].nunique():,}")
    c3.metric("Recent Window", f"{recent_days} days")
    c4.metric("Prev Window", f"{prev_days} days")
    st.caption(f"Window: {start_recent.date()} → {end_date.date()} (prev: {start_prev.date()} → {start_recent.date()})")

    st.subheader("Sample rows")
    st.dataframe(df[["publishedAt_local","title","tags_list"]].head(10), use_container_width=True)

with tab2:
    st.subheader("Top Trending Keywords (micro)")
    top_trends = trend.sort_values(["rank_score","count_recent"], ascending=[False, False]).head(20).reset_index(drop=True)
    st.dataframe(top_trends, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8,6))
    t10 = top_trends[["token","rank_score"]].iloc[::-1]
    ax.barh(t10["token"], t10["rank_score"])
    ax.set_title("Top 20 Trending Keywords")
    ax.set_xlabel("Rank Score (growth-adjusted)")
    st.pyplot(fig)

with tab3:
    st.subheader("Momentum (engagement-weighted)")
    top_m = trend_m.sort_values("momentum_score", ascending=False).head(20).reset_index(drop=True)
    st.dataframe(top_m, use_container_width=True)
    fig, ax = plt.subplots(figsize=(8,6))
    m10 = top_m[["token","momentum_score"]].iloc[::-1]
    ax.barh(m10["token"], m10["momentum_score"])
    ax.set_title("Top 20 by Momentum")
    ax.set_xlabel("Momentum Score")
    st.pyplot(fig)

with tab4:
    st.subheader("Segment Distribution Across Top Trends")
    exploded = ensure_exploded_with_title(df)
    top_tokens = set(trend.sort_values("rank_score", ascending=False).head(10)["token"])
    seg = exploded[exploded["token"].isin(top_tokens)].copy()
    seg["segment"] = seg["topicCategories"].apply(map_segment)
    overall = seg.groupby("segment").size().reset_index(name="videos").sort_values("videos", ascending=False)
    st.dataframe(overall, use_container_width=True)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(overall["segment"], overall["videos"])
    ax.set_title("Segments (Top 10 Trends)")
    ax.set_xlabel("Segment"); ax.set_ylabel("Videos")
    st.pyplot(fig)

with tab5:
    st.subheader("Decay Explorer")
    token_choice = st.selectbox("Pick a token", sorted(trend["token"].unique()))
    series = (weekly_counts[weekly_counts["token"] == token_choice]
              .sort_values("week").set_index("week")["count"]
              .asfreq("W-MON", fill_value=0))
    rolling = series.rolling(3, min_periods=1).mean()
    peak = rolling.max()
    decay_point = None
    below = rolling < (0.6 * peak)
    if peak > 0:
        peaked_idx = rolling.idxmax()
        after_peak = rolling[rolling.index > peaked_idx]
        for ts in after_peak.index:
            if below.get(ts, False) and below.shift(1).get(ts, False):
                decay_point = ts
                break
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(series.index, series.values, label="Weekly Count")
    ax.plot(rolling.index, rolling.values, label="3-Week Rolling Avg")
    if decay_point is not None:
        ax.axvline(decay_point, linestyle="--", label=f"Decay Point ({decay_point.date()})")
    ax.set_title(f"Trend Decay — '{token_choice}'")
    ax.set_xlabel("Week"); ax.set_ylabel("Mentions"); ax.legend()
    st.pyplot(fig)

    if HAS_SM and len(series) >= 6:
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None).fit()
            forecast = model.forecast(4)
            fig2, ax2 = plt.subplots(figsize=(9,4))
            ax2.plot(series.index, series.values, label="Weekly Count")
            ax2.plot(forecast.index, forecast.values, label="Forecast", linestyle="--")
            ax2.set_title(f"Decay Forecast — '{token_choice}'")
            ax2.legend()
            st.pyplot(fig2)
        except Exception as e:
            st.info(f"Forecasting skipped: {e}")

with tab6:
    st.subheader("Semantic Topics (Transformer → KMeans; TF-IDF fallback)")
    with st.spinner("Building topics..."):
        top_topics, df_recent = dl_topics(df, recent_days, prev_days)
    if top_topics is None or df_recent is None or df_recent.empty:
        st.warning("Semantic topics unavailable (need >=50 recent titles and scikit-learn).")
    else:
        st.dataframe(top_topics, use_container_width=True)
        fig, ax = plt.subplots(figsize=(8,6))
        t10 = top_topics[["topic_label","rank_score"]].iloc[::-1]
        ax.barh(t10["topic_label"], t10["rank_score"])
        ax.set_title("Top Semantic Topics (Last 30 Days)")
        ax.set_xlabel("Rank Score")
        st.pyplot(fig)

with tab7:
    st.subheader("Hybrid Ranking (micro × macro × momentum)")
    with st.spinner("Linking tokens ↔ topics and blending..."):
        exploded = ensure_exploded_with_title(df)
        top_topics, df_recent = dl_topics(df, recent_days, prev_days)
        if top_topics is None or df_recent is None or df_recent.empty:
            st.warning("Hybrid needs semantic topics — see the previous tab first.")
        else:
            hybrid = build_hybrid_simple(trend, trend_m, top_topics, df_recent, exploded, alpha=alpha, beta=beta)
            if hybrid is None or hybrid.empty:
                st.warning("No hybrid results.")
            else:
                st.dataframe(hybrid.head(20)[["token","mapped_topic","token_score","topic_score","mom_score","hybrid_score"]],
                             use_container_width=True)
                fig, ax = plt.subplots(figsize=(10,6))
                h10 = hybrid.head(15)[["token","hybrid_score"]].iloc[::-1]
                ax.barh(h10["token"], h10["hybrid_score"])
                ax.set_title("Hybrid Score (Top 15)")
                ax.set_xlabel("Hybrid Score")
                st.pyplot(fig)

with tab8:  # NEW: Audio-Language
    st.subheader("Audio-Language Views (All-Time Distribution & Token Mix)")
    exploded = ensure_exploded_with_title(df)
    top_tokens_list = trend.sort_values("rank_score", ascending=False).head(10)["token"].tolist()
    aud_total, cross, matrix, matrix_pct = audio_language_views(exploded, top_tokens_list)

    st.write("**All-time audio-language distribution**")
    st.dataframe(aud_total.head(20), use_container_width=True)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.barh(aud_total.head(10)["audio_lang"], aud_total.head(10)["videos"])
    ax.set_title("Top Audio Languages (All-Time)")
    ax.set_xlabel("Number of Videos")
    st.pyplot(fig)

    st.write("**Token × Audio-Language crosstab (top tokens)**")
    st.dataframe(matrix.head(10), use_container_width=True)

    token_choice2 = st.selectbox("Pick a token for language mix", sorted(set(top_tokens_list)))
    mix = (cross[cross["token"]==token_choice2].sort_values("videos", ascending=False).head(10))
    st.write(f"Audio-language mix for **{token_choice2}**")
    st.dataframe(mix, use_container_width=True)
    fig2, ax2 = plt.subplots(figsize=(7,4))
    ax2.barh(mix["audio_lang"], mix["videos"])
    ax2.set_title(f"Audio-Language Mix — '{token_choice2}'")
    ax2.set_xlabel("Videos")
    st.pyplot(fig2)

with tab9:  # NEW: Evaluation
    st.subheader("Evaluation (Beauty Precision@10 & Lead-Time)")
    # Precision@10 on raw token trend
    top_tokens_ordered = trend.sort_values(["rank_score","count_recent"], ascending=[False, False])["token"].tolist()
    p10 = precision_at_k(top_tokens_ordered, 10)
    st.metric("Beauty Precision@10 (raw)", f"{p10:.2f}" if not np.isnan(p10) else "N/A")

    # Lead time for the #1 token (raw and last-12m adjusted)
    if len(top_tokens_ordered) > 0:
        best_token = top_tokens_ordered[0]
        raw_w, adj_w = lead_time_weeks(weekly_counts, best_token, lookback_months=12)
        st.write(f"**Lead time for '{best_token}':** raw **{raw_w:.1f} wks**, last-12m **{adj_w:.1f} wks**" if not (np.isnan(raw_w) or np.isnan(adj_w)) else "Lead time: N/A")
    else:
        st.write("Lead time: N/A")

st.caption("© 2025 TrendSpotter — Prototype for L'Oréal × Monash Datathon")
