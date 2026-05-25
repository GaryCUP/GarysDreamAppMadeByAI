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
    st.subheader("Dreams per day")
    daily = df.groupby("dream_date").size().reset_index(name="count")
    if not daily.empty:
        fig_day = px.line(daily, x="dream_date", y="count", markers=True, title="Dreams per day")
        fig_day.update_layout(xaxis_title="Date")
        st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.write("No date information available.")

with t2:
    st.subheader("Tag frequency over time (cumulative)")
    suggested = top_tags["tag"].head(12).tolist() if not top_tags.empty else []
    
    # Get all unique tags for autocomplete
    all_tags = set()
    for _, r in df.iterrows():
        tb = r.get("tags_by_type", {}) or {}
        if sel_type == "ALL":
            for v in tb.values():
                all_tags.update(v)
        else:
            all_tags.update(tb.get(sel_type, []))
    all_tags_sorted = sorted(list(all_tags))
    
    # Allow users to select from suggestions or add custom tags
    selected_tags = st.multiselect(
        "Select tags to show (suggested top tags + custom tags)", 
        all_tags_sorted,
        default=suggested[:6],
        help="Start typing to filter or add custom tags",
        key="temporal_tags"
    )
    
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
                    rows.append((r["dream_date"], t))
        tt = pd.DataFrame(rows, columns=["date","tag"]) if rows else pd.DataFrame(columns=["date","tag"])
        if not tt.empty:
            heat = tt.groupby(["date","tag"]).size().reset_index(name="count")
            pivot = heat.pivot(index="date", columns="tag", values="count").fillna(0)
            # Sort by date to ensure correct cumulative calculation
            pivot = pivot.sort_index()
            # Calculate cumulative sum
            cumulative = pivot.cumsum()
            
            # For each tag, only show data starting from first occurrence
            cumulative_trimmed = cumulative.copy()
            for col in cumulative_trimmed.columns:
                first_occurrence = (cumulative[col] > 0).idxmax()
                cumulative_trimmed.loc[:first_occurrence, col] = None
            
            # Create line chart manually with individual traces for better hover control
            fig_line = go.Figure()
            for tag in cumulative_trimmed.columns:
                data = cumulative_trimmed[tag].dropna()
                fig_line.add_trace(go.Scatter(
                    x=data.index,
                    y=data.values,
                    mode='lines+markers',
                    name=tag,
                    hovertemplate=f'<b>{tag}</b><br>Date: %{{x}}<br>Count: %{{y}}<extra></extra>'
                ))
            
            fig_line.update_layout(
                title="Tag frequency over time (cumulative)",
                xaxis_title="Date",
                yaxis_title="Cumulative Count",
                hovermode="x"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No occurrences for selected tags in the data.")
# -------------------------
# Heatmap: Day-of-week
# -------------------------
st.header("Day-of-week heatmap")
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

# Get all unique tags for selection
all_tags_for_heatmap = set()
for _, r in df.iterrows():
    tb = r.get("tags_by_type", {}) or {}
    if sel_type == "ALL":
        for v in tb.values():
            all_tags_for_heatmap.update(v)
    else:
        all_tags_for_heatmap.update(tb.get(sel_type, []))
all_tags_for_heatmap = sorted(list(all_tags_for_heatmap))

# Default to top 10 for initial display
top10_for_type = flatten_tags_by_type(df, sel_type)["tag"].value_counts().head(10).index.tolist() if not flat_for_sel.empty else []

# Allow users to select any tags
selected_heatmap_tags = st.multiselect(
    "Select tags for heatmap (or add custom tags)",
    all_tags_for_heatmap,
    default=top10_for_type,
    key="heatmap_tags",
    help="Start typing to filter or select any available tags"
)

heat_tbl = build_weekday_table_for_type(selected_heatmap_tags, sel_type)
if not heat_tbl.empty:
    fig_heat = px.imshow(heat_tbl.values, x=heat_tbl.columns, y=heat_tbl.index, aspect="auto", title="Tag by weekday heatmap")
    st.plotly_chart(fig_heat, use_container_width=True)
else:
    st.write("Not enough data for heatmap with selected tags.")

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
    
    # Get all unique tags that appear in edges
    all_network_tags = sorted(list(set(edge_df["tag1"].tolist() + edge_df["tag2"].tolist())))
    
    # Allow users to filter the network by selecting specific tags
    selected_network_tags = st.multiselect(
        "Select tags to include in network (leave empty for all)",
        all_network_tags,
        key="network_tags",
        help="Filter network to show only edges involving selected tags"
    )
    
    # Filter by selected tags
    if selected_network_tags:
        filtered_edges = edge_df[
            (edge_df["tag1"].isin(selected_network_tags)) | 
            (edge_df["tag2"].isin(selected_network_tags))
        ]
    else:
        filtered_edges = edge_df
    
    # Option to filter by minimum co-occurrence weight
    min_weight = st.slider("Minimum co-occurrences to show", 1, int(filtered_edges["weight"].max()), 1)
    top_edges = filtered_edges[filtered_edges["weight"] >= min_weight]
    
    G = nx.Graph()
    for _, r in top_edges.iterrows():
        G.add_edge(r["tag1"], r["tag2"], weight=int(r["weight"]))
    
    if len(G.nodes()) > 0:
        node_sizes = {t: (1 + (tag_counts.get(t,0) if not tag_counts.empty else 0)) for t in G.nodes()}
        
        # Better layout parameters to spread out the network
        pos = nx.spring_layout(G, k=2, iterations=100, seed=42, scale=2)
        
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
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                                line=dict(width=0.5, color='rgba(125,125,125,0.5)'), 
                                hoverinfo='none', name='')
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', 
                                text=[n for n in G.nodes()], 
                                textposition="top center",
                                textfont=dict(size=10),
                                hovertext=text, 
                                marker=dict(size=size, color='lightblue', line=dict(width=2, color='darkblue')),
                                name='')
        
        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            title=f"Tag co-occurrence network ({len(G.nodes())} nodes, {len(G.edges())} edges)",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700,
            plot_bgcolor='rgba(240,240,240,0.9)'
        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.write("No nodes to display with selected filters.")
else:
    st.write("No co-occurring tags found for selected type.")
# -------------------------
# NEW: Degrees of Separation
# -------------------------
st.header("🔗 Degrees of Separation")
st.markdown("Explore how closely connected any two tags are through intermediate tags.")

if edge_list:
    # Build full graph from edge list
    G_full = nx.Graph()
    for _, r in edge_df.iterrows():
        G_full.add_edge(r["tag1"], r["tag2"], weight=int(r["weight"]))
    
    col1, col2 = st.columns(2)
    all_tags = sorted(G_full.nodes())
    
    with col1:
        tag_a = st.selectbox("From tag", all_tags, key="dos_from")
    with col2:
        tag_b = st.selectbox("To tag", [t for t in all_tags if t != tag_a], key="dos_to")
    
    if st.button("Find Connection Path"):
        try:
            if nx.has_path(G_full, tag_a, tag_b):
                path = nx.shortest_path(G_full, tag_a, tag_b)
                degrees = len(path) - 1
                
                st.success(f"**{degrees} degrees of separation** between '{tag_a}' and '{tag_b}'")
                st.write("**Path:** " + " → ".join(path))
                
                # Visualize the path
                path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
                G_path = nx.Graph()
                G_path.add_edges_from(path_edges)
                
                pos_path = nx.spring_layout(G_path, k=2, seed=42)
                
                # Draw edges
                edge_x_path = []
                edge_y_path = []
                for e in G_path.edges():
                    x0, y0 = pos_path[e[0]]
                    x1, y1 = pos_path[e[1]]
                    edge_x_path += [x0, x1, None]
                    edge_y_path += [y0, y1, None]
                
                # Draw nodes
                node_x_path = []
                node_y_path = []
                node_text = []
                node_color = []
                for i, n in enumerate(path):
                    x, y = pos_path[n]
                    node_x_path.append(x)
                    node_y_path.append(y)
                    node_text.append(n)
                    if i == 0:
                        node_color.append('green')
                    elif i == len(path) - 1:
                        node_color.append('red')
                    else:
                        node_color.append('orange')
                
                edge_trace_path = go.Scatter(x=edge_x_path, y=edge_y_path, mode='lines',
                                            line=dict(width=3, color='#888'), hoverinfo='none')
                node_trace_path = go.Scatter(x=node_x_path, y=node_y_path, mode='markers+text',
                                            text=node_text, textposition="top center",
                                            marker=dict(size=25, color=node_color, line=dict(width=2, color='white')))
                
                fig_path = go.Figure(data=[edge_trace_path, node_trace_path])
                fig_path.update_layout(title=f"Connection Path: {tag_a} → {tag_b}",
                                      showlegend=False, height=400,
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                st.plotly_chart(fig_path, use_container_width=True)
                
                # Show all shortest paths if multiple exist
                all_paths = list(nx.all_shortest_paths(G_full, tag_a, tag_b))
                if len(all_paths) > 1:
                    st.info(f"Found {len(all_paths)} equally short paths (showing first):")
                    for i, p in enumerate(all_paths[:5]):
                        st.write(f"{i+1}. " + " → ".join(p))
            else:
                st.warning(f"No connection found between '{tag_a}' and '{tag_b}' in the current network.")
        except Exception as e:
            st.error(f"Error finding path: {e}")
    
    # Compute average degrees of separation
    st.subheader("Network Connectivity Statistics")
    if nx.is_connected(G_full):
        avg_path_length = nx.average_shortest_path_length(G_full)
        diameter = nx.diameter(G_full)
        st.metric("Average Path Length", f"{avg_path_length:.2f}")
        st.metric("Network Diameter", diameter)
        st.caption("Average path length = average degrees of separation between all tag pairs")
    else:
        components = list(nx.connected_components(G_full))
        st.info(f"Network has {len(components)} disconnected components. Showing stats for largest:")
        largest = max(components, key=len)
        G_largest = G_full.subgraph(largest)
        avg_path = nx.average_shortest_path_length(G_largest)
        diam = nx.diameter(G_largest)
        st.metric("Avg Path Length (largest component)", f"{avg_path:.2f}")
        st.metric("Diameter (largest component)", diam)

else:
    st.write("Not enough co-occurrence data to compute degrees of separation.")

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

# -------------------------
# NEW: Advanced Markov Chain Analysis
# -------------------------
st.header("🎲 Markov Chain Modeling & Prediction")
st.markdown("Build transition probability matrices and predict likely next tags.")

# Build transition pairs
pairs = []
for _, r in df.iterrows():
    tb = r.get("tags_by_type", {}) or {}
    seq = []
    if sel_type == "ALL":
        for k in sorted(tb.keys()):
            seq += tb.get(k, [])
    else:
        seq = tb.get(sel_type, [])
    for i in range(len(seq)-1):
        pairs.append((seq[i], seq[i+1]))

if pairs:
    pairs_df = pd.DataFrame(pairs, columns=["from","to"])
    trans = pairs_df.groupby(["from","to"]).size().reset_index(name="count")
    trans["prob"] = trans.groupby("from")["count"].transform(lambda s: s / s.sum())
    
    st.subheader("Transition Probability Matrix")
    st.write("First-order Markov transitions (from → to):")
    st.dataframe(trans.sort_values("prob", ascending=False).head(100))
    
    # Build transition matrix
    all_states = sorted(set(trans["from"].unique()) | set(trans["to"].unique()))
    trans_matrix = pd.DataFrame(0.0, index=all_states, columns=all_states)
    
    for _, row in trans.iterrows():
        trans_matrix.loc[row["from"], row["to"]] = row["prob"]
    
    # Interactive predictor
    st.subheader("🔮 Tag Sequence Predictor")
    st.markdown("Enter current tags and see predicted next tags based on Markov transitions.")
    
    current_tag = st.selectbox("Current tag", all_states, key="markov_current")
    
    if st.button("Predict Next Tags"):
        next_probs = trans_matrix.loc[current_tag]
        next_probs = next_probs[next_probs > 0].sort_values(ascending=False)
        
        if len(next_probs) > 0:
            st.success(f"**Top predicted tags after '{current_tag}':**")
            
            pred_df = pd.DataFrame({
                'Next Tag': next_probs.index,
                'Probability': next_probs.values,
                'Percentage': (next_probs.values * 100).round(1)
            })
            
            st.dataframe(pred_df.head(15))
            
            # Visualize predictions
            fig_pred = px.bar(pred_df.head(10), x='Next Tag', y='Percentage',
                             title=f"Probability Distribution of Next Tags after '{current_tag}'")
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.warning(f"No recorded transitions from '{current_tag}'")
    
    # Multi-step prediction
    st.subheader("🎯 Multi-Step Sequence Prediction")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        start_tag = st.selectbox("Starting tag", all_states, key="markov_start")
    with col2:
        n_steps = st.slider("Steps ahead", 1, 10, 3)
    
    if st.button("Generate Sequence"):
        sequence = [start_tag]
        current = start_tag
        
        for step in range(n_steps):
            next_probs = trans_matrix.loc[current]
            next_probs = next_probs[next_probs > 0]
            
            if len(next_probs) == 0:
                st.info(f"Chain stopped at step {step+1}: no transitions from '{current}'")
                break
            
            # Choose most likely next tag
            next_tag = next_probs.idxmax()
            sequence.append(next_tag)
            current = next_tag
        
        st.success("**Predicted Sequence:**")
        st.write(" → ".join(sequence))
        
        # Show probabilities at each step
        st.write("**Transition Probabilities:**")
        for i in range(len(sequence)-1):
            prob = trans_matrix.loc[sequence[i], sequence[i+1]]
            st.write(f"Step {i+1}: {sequence[i]} → {sequence[i+1]} ({prob:.1%})")
    
    # Stationary distribution
    st.subheader("📊 Stationary Distribution")
    st.markdown("Long-term probability of being in each state (if chain converges)")
    
    try:
        # Convert to numpy for eigenvalue calculation
        P = trans_matrix.values
        if P.shape[0] > 0 and np.any(P > 0):
            # Find stationary distribution via eigenvector
            eigenvalues, eigenvectors = np.linalg.eig(P.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary = np.real(eigenvectors[:, stationary_idx])
            stationary = stationary / stationary.sum()
            
            stat_df = pd.DataFrame({
                'Tag': all_states,
                'Stationary Probability': stationary,
                'Percentage': (stationary * 100).round(2)
            }).sort_values('Stationary Probability', ascending=False)
            
            st.dataframe(stat_df.head(20))
            
            fig_stat = px.bar(stat_df.head(15), x='Tag', y='Percentage',
                            title="Stationary Distribution (Top 15 Tags)")
            st.plotly_chart(fig_stat, use_container_width=True)
            
            st.caption("Stationary distribution shows the long-run probability of each tag appearing in the sequence.")
    except Exception as e:
        st.info(f"Could not compute stationary distribution: {e}")
    
    # Sankey diagram for transitions
    st.subheader("Transition Flow Diagram (Sankey)")
    node_limit = st.slider("Max distinct tags in Sankey", 5, 60, 30)
    top_nodes = pd.concat([trans["from"], trans["to"]]).value_counts().head(node_limit).index.tolist()
    trans_sank = trans[trans["from"].isin(top_nodes) & trans["to"].isin(top_nodes)]
    
    if not trans_sank.empty:
        label_map = {n:i for i,n in enumerate(sorted(top_nodes))}
        sources = trans_sank["from"].map(label_map).tolist()
        targets = trans_sank["to"].map(label_map).tolist()
        values = trans_sank["count"].tolist()
        sankey = go.Sankey(node=dict(label=list(sorted(top_nodes))), 
                          link=dict(source=sources, target=targets, value=values))
        fig_sank = go.Figure(sankey)
        fig_sank.update_layout(title="Tag transition Sankey (limited nodes)", height=600)
        st.plotly_chart(fig_sank, use_container_width=True)
    
    # Monte Carlo simulation
    st.subheader("🎲 Monte Carlo Simulation")
    st.markdown("Generate random dream sequences using the learned transition probabilities")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        sim_start = st.selectbox("Starting tag for simulation", all_states, key="sim_start")
    with col2:
        sim_length = st.slider("Sequence length", 3, 20, 8)
    with col3:
        n_simulations = st.slider("# Simulations", 1, 10, 3)
    
    if st.button("Run Simulation"):
        st.write("**Generated Dream Sequences:**")
        
        for sim_num in range(n_simulations):
            sequence = [sim_start]
            current = sim_start
            
            for step in range(sim_length - 1):
                next_probs = trans_matrix.loc[current]
                next_probs = next_probs[next_probs > 0]
                
                if len(next_probs) == 0:
                    break
                
                # Randomly sample based on probabilities
                next_tag = np.random.choice(next_probs.index, p=next_probs.values)
                sequence.append(next_tag)
                current = next_tag
            
            st.write(f"**Simulation {sim_num + 1}:** " + " → ".join(sequence))
        
        st.caption("Each simulation randomly samples from the transition probabilities, creating possible dream tag sequences.")

else:
    st.write("Not enough sequential data for Markov modeling for selected type.")

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

st.caption("Built with 🤖, by AI")
