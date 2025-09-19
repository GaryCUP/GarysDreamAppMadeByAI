# dream_dashboard_full_adapted.py
import streamlit as st
import pandas as pd
import numpy as np
from itertools import combinations
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import io, math
from datetime import timedelta

st.set_page_config(page_title="Dream Journal Analytics — Full (tag_type)", layout="wide", initial_sidebar_state="expanded")

st.title("🌙 Dream Journal Analytics — Full (preserves tag_type)")
st.markdown("""
Upload a CSV. The app supports two common schemas:

**Simple** (one row per dream)  
- Columns: `date`, `tags` (comma-separated), optional `text`/`description`/`title`.

**Normalized** (one row per tag)  
- Columns: `id` (or `dream_id`), `date`, `tag_name`, `tag_type`, optional `title`, `description`.

This app preserves `tag_type` and allows filtering / analysis by tag type.
""")

# --- Sidebar: upload & settings ---
st.sidebar.header("Upload & settings")
uploaded = st.sidebar.file_uploader("Upload dream journal CSV", type=["csv"])
use_sample = False
if uploaded is None:
    st.sidebar.info("No file uploaded — you can press \"Use sample\" if a sample exists in the workspace.")
    try:
        sample_path = "/mnt/data/elsewhere-dreams-2025-04-09.csv"
        with open(sample_path, "rb"):
            if st.sidebar.button("Use sample file"):
                uploaded = sample_path
                use_sample = True
    except Exception:
        pass

if uploaded is None:
    st.stop()

# --- Load CSV ---
if use_sample:
    df_raw = pd.read_csv(uploaded, parse_dates=["date"], low_memory=False)
else:
    df_raw = pd.read_csv(uploaded, parse_dates=["date"], low_memory=False)

st.sidebar.markdown(f"Rows: {len(df_raw):,} | Columns: {', '.join(df_raw.columns)}")

# -------------------------
# Parsing helpers
# -------------------------
def parse_simple_schema(df):
    """Expect a 'tags' column with comma-separated tags. Return one-row-per-dream dataframe."""
    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    # detect text field
    text_col = None
    for c in ["text", "description", "note", "body"]:
        if c in df2.columns:
            text_col = c
            break
    df2["text"] = df2[text_col] if text_col is not None else ""
    def split_tags(x):
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return [str(t).strip() for t in x if str(t).strip()]
        return [t.strip() for t in str(x).split(",") if t.strip()]
    df2["tags_all"] = df2.get("tags", "").apply(split_tags)
    df2["tags_by_type"] = df2["tags_all"].apply(lambda lst: {"ALL": lst})
    out = df2[["date", "text", "tags_all", "tags_by_type"]].rename(columns={"tags_all": "tags"})
    # ensure columns exist
    out["id"] = out.index.astype(str)
    return out[["id","date","text","tags","tags_by_type"]]

def parse_normalized_schema(df):
    """
    Expect repeated rows per dream with tag_name + tag_type.
    Group by id / dream_id (fall back to row index) and aggregate tags per type.
    """
    df2 = df.copy()
    if "date" in df2.columns:
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    # find id column
    id_col = None
    for c in ["id","dream_id","entry_id"]:
        if c in df2.columns:
            id_col = c
            break
    if id_col is None:
        df2["_row_group"] = df2.index.astype(str)
        id_col = "_row_group"
    # possible title/text columns
    title_col = None
    for c in ["title","summary"]:
        if c in df2.columns:
            title_col = c
            break
    text_col = None
    for c in ["description","text","note","body"]:
        if c in df2.columns:
            text_col = c
            break
    # require tag columns
    if "tag_name" not in df2.columns or "tag_type" not in df2.columns:
        raise ValueError("Normalized schema requires 'tag_name' and 'tag_type' columns.")
    grouped = []
    for gid, g in df2.groupby(id_col):
        date = g["date"].dropna().min() if "date" in g.columns else pd.NaT
        title = g[title_col].dropna().iloc[0] if title_col and g[title_col].dropna().shape[0] > 0 else ""
        text = g[text_col].dropna().iloc[0] if text_col and g[text_col].dropna().shape[0] > 0 else ""
        tags_by_type = {}
        for _, row in g.iterrows():
            ttype = row.get("tag_type", "UNKNOWN")
            tname = row.get("tag_name", "")
            if pd.isna(tname) or tname == "":
                continue
            ttype = str(ttype)
            tags_by_type.setdefault(ttype, []).append(str(tname))
        # dedupe and sort per type
        tags_by_type = {k: sorted(set(v)) for k,v in tags_by_type.items()}
        flat = [t for v in tags_by_type.values() for t in v]
        grouped.append({"id": gid, "date": date, "title": title, "text": text, "tags_by_type": tags_by_type, "tags": flat})
    out = pd.DataFrame(grouped)
    return out[["id","date","title","text","tags","tags_by_type"]]

# auto-detect parser
try:
    if "tag_name" in df_raw.columns and "tag_type" in df_raw.columns:
        parsed = parse_normalized_schema(df_raw)
        used_schema = "normalized"
    elif "tags" in df_raw.columns:
        parsed = parse_simple_schema(df_raw)
        used_schema = "simple"
    else:
        # attempt to synthesize tags from any 'tag*' columns
        tag_cols = [c for c in df_raw.columns if c.lower().startswith("tag")]
        if tag_cols and ("tag_name" in tag_cols or "tags" in tag_cols):
            if "tag_name" in df_raw.columns and "tag_type" in df_raw.columns:
                parsed = parse_normalized_schema(df_raw)
                used_schema = "normalized"
            elif "tags" in df_raw.columns:
                parsed = parse_simple_schema(df_raw)
                used_schema = "simple"
            else:
                st.error("Could not find clear tags schema. Please ensure 'tags' or 'tag_name'+'tag_type' exist.")
                st.stop()
        else:
            st.error("Couldn't detect a supported schema. Needs 'tags' column (simple) OR 'tag_name' + 'tag_type' (normalized).")
            st.stop()
except Exception as e:
    st.error(f"Failed to parse file: {e}")
    st.stop()

st.sidebar.success(f"Parsed using: {used_schema} schema")

# ensure standardized columns
df = parsed.copy()
if "tags_by_type" not in df.columns:
    df["tags_by_type"] = df["tags_by_type"].apply(lambda x: {"ALL": x} if isinstance(x, list) else x)
if "tags" not in df.columns:
    df["tags"] = df["tags_by_type"].apply(lambda d: [t for v in d.values() for t in v])

# derive helpers
df = df.sort_values("date").reset_index(drop=True)
df["year_month"] = df["date"].dt.to_period("M").astype(str)
df["weekday"] = df["date"].dt.day_name()
df["tag_count"] = df["tags"].apply(lambda x: len(x) if isinstance(x, list) else 0)

def flatten_tags_by_type(df, ttype="ALL"):
    rows = []
    for _, r in df.iterrows():
        tb = r.get("tags_by_type", {}) or {}
        if ttype == "ALL":
            for k, v in tb.items():
                for t in v:
                    rows.append((t, k, r["date"], r.get("id", "")))
        else:
            for t in tb.get(ttype, []):
                rows.append((t, ttype, r["date"], r.get("id", "")))
    return pd.DataFrame(rows, columns=["tag","tag_type","date","dream_id"])

flat_all = flatten_tags_by_type(df, "ALL")
tag_counts = flat_all["tag"].value_counts() if not flat_all.empty else pd.Series([], dtype=int)
unique_types = sorted(flat_all["tag_type"].dropna().unique()) if not flat_all.empty else []

# -------------------------
# Layout: Overview & Top
# -------------------------
st.header("Overview")
o1, o2, o3, o4 = st.columns([1,1,1,1])
o1.metric("Total dreams", len(df))
date_min = df['date'].min() if 'date' in df.columns else pd.NaT
date_max = df['date'].max() if 'date' in df.columns else pd.NaT
o2.metric("Date range", f"{date_min.date() if pd.notna(date_min) else 'N/A'} → {date_max.date() if pd.notna(date_max) else 'N/A'}")
o3.metric("Unique tags (all types)", int(tag_counts.size if not tag_counts.empty else 0))
o4.metric("Tag types", ", ".join(unique_types) if unique_types else "N/A")

st.subheader("Top tags (choose type)")
sel_type = st.selectbox("Tag type to analyze", ["ALL"] + unique_types)
topk = st.slider("Top k tags to show", 5, 50, 15)
flat_for_sel = flatten_tags_by_type(df, sel_type)
top_tags = flat_for_sel["tag"].value_counts().head(topk).reset_index()
top_tags.columns = ["tag","count"]
if not top_tags.empty:
    st.plotly_chart(px.bar(top_tags, x="tag", y="count", title=f"Top {topk} tags (type={sel_type})"), use_container_width=True)
else:
    st.info("No tags found for selected type.")

# -------------------------
# Temporal Trends
# -------------------------
st.header("Temporal Trends")
t1, t2 = st.columns(2)

with t1:
    st.subheader("Dreams per month")
    monthly = df.groupby("year_month").size().reset_index(name="count")
    if not monthly.empty:
        fig_month = px.line(monthly, x="year_month", y="count", markers=True, title="Dreams per month")
        fig_month.update_layout(xaxis_title="Month")
        st.plotly_chart(fig_month, use_container_width=True)
    else:
        st.write("No date information available.")

with t2:
    st.subheader("Tag frequency over time (stacked area)")
    suggested = top_tags["tag"].head(12).tolist() if not top_tags.empty else []
    selected_tags = st.multiselect("Select tags to show (suggested top tags)", suggested, default=suggested[:6])
    if selected_tags:
        rows = []
        for _, r in df.iterrows():
            tb = r.get("tags_by_type", {}) or {}
            candidates = []
            if sel_type == "ALL":
                for v in tb.values():
                    candidates += v
            else:
                candidates = tb.get(sel_type, [])
            for t in candidates:
                if t in selected_tags:
                    rows.append((r["year_month"], t))
        tt = pd.DataFrame(rows, columns=["month","tag"]) if rows else pd.DataFrame(columns=["month","tag"])
        if not tt.empty:
            heat = tt.groupby(["month","tag"]).size().reset_index(name="count")
            pivot = heat.pivot(index="month", columns="tag", values="count").fillna(0)
            st.area_chart(pivot)
        else:
            st.info("No occurrences for selected tags in the data.")

# -------------------------
# Heatmap: Day-of-week
# -------------------------
st.header("Day-of-week heatmap (top tags)")
def build_weekday_table_for_type(tags_list, ttype):
    rows = []
    for _, r in df.iterrows():
        tb = r.get("tags_by_type", {}) or {}
        if ttype == "ALL":
            for k,v in tb.items():
                for t in v:
                    if t in tags_list:
                        rows.append((t, r["weekday"]))
        else:
            for t in tb.get(ttype, []):
                if t in tags_list:
                    rows.append((t, r["weekday"]))
    tt = pd.DataFrame(rows, columns=["tag","weekday"]) if rows else pd.DataFrame()
    if tt.empty:
        return pd.DataFrame()
    table = tt.groupby(["tag","weekday"]).size().unstack(fill_value=0)
    weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    cols_present = [d for d in weekdays if d in table.columns]
    return table[cols_present]

top10_for_type = flatten_tags_by_type(df, sel_type)["tag"].value_counts().head(10).index.tolist() if not flat_for_sel.empty else []
heat_tbl = build_weekday_table_for_type(top10_for_type, sel_type)
if not heat_tbl.empty:
    fig_heat = px.imshow(heat_tbl.values, x=heat_tbl.columns, y=heat_tbl.index, aspect="auto", title="Tag by weekday heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("Not enough data for heatmap for the selected type.")

# -------------------------
# Co-occurrence Network
# -------------------------
st.header("Tag co-occurrence network")
edge_list = []
for _, r in df.iterrows():
    tb = r.get("tags_by_type", {}) or {}
    if sel_type == "ALL":
        candidates = [t for v in tb.values() for t in v]
    else:
        candidates = tb.get(sel_type, [])
    for a,b in combinations(sorted(set(candidates)), 2):
        edge_list.append((a,b))
if edge_list:
    edge_df = pd.Series(edge_list).value_counts().reset_index(name="weight")
    edge_df[["tag1","tag2"]] = pd.DataFrame(edge_df["index"].tolist(), index=edge_df.index)
    edge_df = edge_df.drop(columns=["index"])
    max_edges = st.slider("Max edges to show in network", 20, 500, 120)
    top_edges = edge_df.head(max_edges)
    G = nx.Graph()
    for _, r in top_edges.iterrows():
        G.add_edge(r["tag1"], r["tag2"], weight=int(r["weight"]))
    node_sizes = {t: (1 + (tag_counts.get(t,0) if not tag_counts.empty else 0)) for t in G.nodes()}
    pos = nx.spring_layout(G, k=0.5, seed=42)
    edge_x=[]; edge_y=[]
    for e in G.edges():
        x0,y0 = pos[e[0]]
        x1,y1 = pos[e[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    node_x=[]; node_y=[]; text=[]; size=[]
    for n in G.nodes():
        x,y = pos[n]
        node_x.append(x); node_y.append(y)
        text.append(f"{n} ({tag_counts.get(n,0) if not tag_counts.empty else 0})")
        size.append(math.sqrt(node_sizes[n]) * 6 + 6)
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1), hoverinfo='none')
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[n for n in G.nodes()], textposition="top center",
                            hovertext=text, marker=dict(size=size, showscale=False))
    fig_net = go.Figure(data=[edge_trace, node_trace])
    fig_net.update_layout(title="Tag co-occurrence network", showlegend=False,
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600)
    st.plotly_chart(fig_net, use_container_width=True)
else:
    st.write("No co-occurring tags found for selected type.")

# -------------------------
# Complexity & Entropy
# -------------------------
st.header("Complexity & Diversity")
st.subheader("Tags per dream distribution")
fig_hist = px.histogram(df, x="tag_count", nbins=10, title="Tags per dream")
st.plotly_chart(fig_hist, use_container_width=True)

def entropy_of_list(lst):
    if len(lst) == 0:
        return 0.0
    vc = pd.Series(lst).value_counts(normalize=True)
    return -(vc * np.log2(vc)).sum()

entropy_month = df.groupby("year_month")["tags"].apply(lambda s: entropy_of_list([t for sub in s for t in sub])).reset_index(name="entropy")
if not entropy_month.empty:
    st.plotly_chart(px.line(entropy_month, x="year_month", y="entropy", markers=True, title="Tag entropy by month"), use_container_width=True)

# -------------------------
# Recurrence & Markov
# -------------------------
st.header("Recurrence & Prediction")

st.subheader("Average days between occurrences for top tags (selected type)")
rec_rows = []
preview_tags = flatten_tags_by_type(df, sel_type)["tag"].value_counts().head(30).index.tolist()
for tag in preview_tags:
    # dates where tag present in selected type
    dates = df[df["tags"].apply(lambda lst: tag in lst if isinstance(lst, list) else False)]["date"].sort_values().dropna()
    if len(dates) > 1:
        diffs = dates.diff().dt.days.dropna()
        rec_rows.append({"tag": tag, "mean_days": float(diffs.mean()), "median_days": float(diffs.median()), "occurrences": int(len(dates))})
rec_df = pd.DataFrame(rec_rows).sort_values("occurrences", ascending=False)
if not rec_df.empty:
    st.dataframe(rec_df.head(200))
else:
    st.write("Not enough recurrence data for selected type.")

# Markov transitions (first-order) & Sankey
st.subheader("First-order Markov transitions (within-dream tag order)")
pairs = []
# define sequence order: use tags_by_type order if preserved; otherwise alphabetical
for _, r in df.iterrows():
    tb = r.get("tags_by_type", {}) or {}
    seq = []
    if sel_type == "ALL":
        # flatten by tag_type stable order (sorted keys)
        for k in sorted(tb.keys()):
            seq += tb.get(k, [])
    else:
        seq = tb.get(sel_type, [])
    # make pairs within same dream (A -> B if B appears after A in the dream's tag sequence)
    for i in range(len(seq)-1):
        pairs.append((seq[i], seq[i+1]))
if pairs:
    pairs_df = pd.DataFrame(pairs, columns=["from","to"])
    trans = pairs_df.groupby(["from","to"]).size().reset_index(name="count")
    trans["prob"] = trans.groupby("from")["count"].transform(lambda s: s / s.sum())
    st.write("Top transitions (from → to):")
    st.dataframe(trans.sort_values("count", ascending=False).head(80))
    # build Sankey (limit nodes for readability)
    # choose top N nodes by occurrence
    node_limit = st.slider("Max distinct tags in Sankey", 5, 60, 30)
    top_nodes = pd.concat([trans["from"], trans["to"]]).value_counts().head(node_limit).index.tolist()
    trans_sank = trans[trans["from"].isin(top_nodes) & trans["to"].isin(top_nodes)]
    if not trans_sank.empty:
        label_map = {n:i for i,n in enumerate(sorted(top_nodes))}
        sources = trans_sank["from"].map(label_map).tolist()
        targets = trans_sank["to"].map(label_map).tolist()
        values = trans_sank["count"].tolist()
        sankey = go.Sankey(node=dict(label=list(sorted(top_nodes))), link=dict(source=sources, target=targets, value=values))
        fig_sank = go.Figure(sankey)
        fig_sank.update_layout(title="Tag transition Sankey (limited nodes)", height=600)
        st.plotly_chart(fig_sank, use_container_width=True)
else:
    st.write("Not enough sequential data for Markov transitions for selected type.")

# -------------------------
# Search & Examples
# -------------------------
st.header("Search & Examples")
tag_search = st.text_input("Show dreams that include tag (type a tag)")
if tag_search:
    matches = df[df["tags"].apply(lambda tags: tag_search in tags if isinstance(tags, list) else False)]
    st.write(f"Found {len(matches)} dreams with tag '{tag_search}'")
    for i, r in matches.iterrows():
        title = r.get("title", "") or ""
        date_str = r["date"].date() if pd.notna(r.get("date", None)) else ""
        st.markdown(f"**{date_str} — {title} — Tags: {', '.join(r.get('tags', []))}**")
        if r.get("text", ""):
            st.write(r.get("text", "")[:1000])
        st.write("---")

# -------------------------
# Export processed data
# -------------------------
st.header("Export processed data")
if st.button("Prepare processed CSV"):
    buf = io.StringIO()
    out = df.copy()
    # convert tags_by_type to semi-structured string for CSV (type:tag1,tag2;type2:tag3)
    def tbt_to_str(d):
        if not d: return ""
        return ";".join([f"{k}:{','.join(v)}" for k,v in d.items()])
    out["tags_by_type_str"] = out["tags_by_type"].apply(tbt_to_str)
    out.to_csv(buf, index=False)
    st.download_button("Download processed CSV", data=buf.getvalue(), file_name="dreams_processed_full.csv", mime="text/csv")

st.caption("Built with ❤️. If you want additional features (wordclouds, sentiment scoring, clustering exports, or a live link), tell me which and I'll add them.")
