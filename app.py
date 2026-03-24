"""
app.py  —  Semantic Code Similarity Detector
=============================================
Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from preprocessor import tokenize_python, normalize_code, get_token_stats
from embeddings import Word2VecEmbedder, GloVeEmbedder
from similarity import (
    cosine_similarity,
    get_similarity_info,
    pairwise_similarities,
    get_suspicious_pairs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantic Code Similarity Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Typography ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Header gradient ── */
.main-title {
    font-size: 2.4rem; font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 4px;
}
.subtitle {
    color: #64748b; font-size: 1rem; margin-top: 0;
}

/* ── Verdict box ── */
.verdict-box {
    border-radius: 16px; padding: 1.6rem 2rem;
    text-align: center; border: 2px solid;
}
.verdict-title { font-size: 1.3rem; font-weight: 700; margin: 0 0 8px 0; }
.verdict-score { font-size: 3.4rem; font-weight: 800; line-height: 1; }
.verdict-sub   { font-size: 0.85rem; opacity: 0.75; margin-top: 4px; }

/* ── Metric cards ── */
.stat-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 14px 16px; text-align: center;
}
.stat-label { font-size: 0.78rem; color: #64748b; font-weight: 500; text-transform: uppercase; letter-spacing: .5px; }
.stat-value { font-size: 1.8rem; font-weight: 700; color: #1e293b; }

/* ── Token chips ── */
.token-wrap { line-height: 2.1; }
.chip {
    display: inline-block; border-radius: 6px;
    padding: 1px 8px; margin: 2px 2px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
}
.chip-kw  { background:#ede9fe; color:#5b21b6; }
.chip-bi  { background:#dbeafe; color:#1e40af; }
.chip-op  { background:#fce7f3; color:#9d174d; }
.chip-lit { background:#dcfce7; color:#15803d; }
.chip-id  { background:#f1f5f9; color:#334155; }

/* ── Alert banners ── */
.alert-danger  { background:#fef2f2; border-left:4px solid #dc2626; border-radius:8px; padding:12px 16px; margin:6px 0; }
.alert-warning { background:#fff7ed; border-left:4px solid #ea580c; border-radius:8px; padding:12px 16px; margin:6px 0; }
.alert-ok      { background:#f0fdf4; border-left:4px solid #16a34a; border-radius:8px; padding:12px 16px; margin:6px 0; }

/* ── Tab styling ── */
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; padding: 10px 20px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_glove(model_name: str) -> GloVeEmbedder | None:
    """Download and cache GloVe model (runs once per session)."""
    emb = GloVeEmbedder(model_name)
    return emb if emb.load() else None


def _chip(token: str) -> str:
    if token.startswith('KW_'):
        return f'<span class="chip chip-kw">{token}</span>'
    if token.startswith('BUILTIN_'):
        return f'<span class="chip chip-bi">{token}</span>'
    if token.startswith('OP_') or token in ('LPAREN','RPAREN','LBRACKET','RBRACKET','LBRACE','RBRACE','COLON','COMMA','DOT','ARROW','SEMICOLON'):
        return f'<span class="chip chip-op">{token}</span>'
    if token in ('STRING_LITERAL','STR','NUMBER_LITERAL','NUM'):
        return f'<span class="chip chip-lit">{token}</span>'
    return f'<span class="chip chip-id">{token}</span>'


def show_tokens(tokens: list, max_tokens: int = 60):
    shown = tokens[:max_tokens]
    html = '<div class="token-wrap">' + ''.join(_chip(t) for t in shown)
    if len(tokens) > max_tokens:
        html += f'<span style="color:#94a3b8; font-size:.8rem"> … +{len(tokens)-max_tokens} more</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def make_gauge(score: float) -> go.Figure:
    pct  = score * 100
    info = get_similarity_info(score)
    fig  = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': '%', 'font': {'size': 42, 'color': info['color']}},
        title={'text': f"<b>{info['label']}</b>",
               'font': {'size': 16, 'color': info['color']}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickvals': [0, 40, 65, 80, 95, 100],
                'ticktext': ['0', '40', '65', '80', '95', '100'],
                'tickwidth': 1, 'tickcolor': '#94a3b8',
            },
            'bar': {'color': info['color'], 'thickness': 0.28},
            'bgcolor': 'white', 'borderwidth': 2, 'bordercolor': '#e2e8f0',
            'steps': [
                {'range': [0, 40],  'color': '#dcfce7'},
                {'range': [40, 65], 'color': '#fef9c3'},
                {'range': [65, 80], 'color': '#ffedd5'},
                {'range': [80, 95], 'color': '#fee2e2'},
                {'range': [95, 100],'color': '#fca5a5'},
            ],
            'threshold': {
                'line': {'color': info['color'], 'width': 4},
                'thickness': 0.8, 'value': pct,
            },
        }
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    return fig


def make_vector_bar(va: np.ndarray, vb: np.ndarray, dims: int = 20) -> go.Figure:
    d = min(dims, len(va))
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Code A', x=[f'd{i}' for i in range(d)],
                         y=va[:d], marker_color='#6366f1'))
    fig.add_trace(go.Bar(name='Code B', x=[f'd{i}' for i in range(d)],
                         y=vb[:d], marker_color='#f59e0b'))
    fig.update_layout(
        title='Document Vectors — First 20 Dimensions', barmode='group',
        height=280, paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def make_heatmap(sim_df: pd.DataFrame) -> go.Figure:
    labels = list(sim_df.columns)
    z      = sim_df.values
    text   = [[f'{v:.1%}' for v in row] for row in z]
    fig    = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        text=text, texttemplate='%{text}',
        textfont={'size': 11},
        colorscale=[[0.0, '#22c55e'], [0.5, '#fbbf24'],
                    [0.8, '#f97316'], [1.0, '#dc2626']],
        zmin=0, zmax=1,
        colorbar=dict(title='Similarity', tickformat='.0%'),
    ))
    fig.update_layout(
        title='Pairwise Similarity Matrix',
        height=max(420, 80 * len(labels)),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _embed(tokens_list, model_type, w2v_cfg=None, glove_name='glove-wiki-gigaword-50'):
    """Embed a list-of-token-lists; return (list_of_vectors, embedder)."""
    if model_type == 'Word2Vec':
        cfg = w2v_cfg or {}
        emb = Word2VecEmbedder(**cfg)
        emb.train(tokens_list)
        vecs = [emb.get_vector(t) for t in tokens_list]
        return vecs, emb
    else:
        with st.spinner('⏳ Loading GloVe vectors (first use downloads ~66 MB)…'):
            emb = _load_glove(glove_name)
        if emb is None:
            st.error('❌ Could not load GloVe. Check your internet connection.')
            st.stop()
        vecs = [emb.get_vector(t) for t in tokens_list]
        return vecs, emb


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    model_type = st.selectbox(
        'Embedding Model',
        ['Word2Vec', 'GloVe'],
        help=(
            '**Word2Vec** — trains a skip-gram model on the submitted code.\n\n'
            '**GloVe** — uses pre-trained Wikipedia + Gigaword vectors (~66 MB download).'
        ),
    )

    normalize = st.toggle(
        'Normalize Identifiers',
        value=True,
        help='Replace user-defined variable/function names with var_0, var_1 … '
             'Makes the detector robust to trivial renaming.',
    )

    if model_type == 'Word2Vec':
        st.markdown('#### Word2Vec Parameters')
        vec_size = st.select_slider('Vector Size', [50, 100, 150, 200], value=100)
        window   = st.slider('Context Window', 2, 10, 5)
        epochs   = st.select_slider('Epochs', [50, 100, 200, 300, 500], value=200)
        w2v_cfg  = dict(vector_size=vec_size, window=window, epochs=epochs)
    else:
        glove_variant = st.selectbox(
            'GloVe Variant',
            ['glove-wiki-gigaword-50', 'glove-wiki-gigaword-100', 'glove-twitter-25'],
        )
        w2v_cfg = None

    st.markdown('---')
    st.markdown('#### 🚨 Plagiarism Threshold')
    threshold = st.slider(
        'Flag pairs above:', 0.50, 1.00, 0.80, 0.05,
        format='%.2f',
    )

    st.markdown('---')
    st.markdown("""
**Pipeline**
1. **Tokenize** — Python `tokenize` module  
2. **Normalize** — strip identifiers (optional)  
3. **Embed** — Word2Vec or GloVe  
4. **Pool** — mean of token vectors  
5. **Compare** — cosine similarity  
""")

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🔍 Semantic Code Similarity Detector</div>'
    '<p class="subtitle">Logic-Based Plagiarism Checker using Word Embeddings &amp; Cosine Similarity</p>',
    unsafe_allow_html=True,
)
st.markdown('---')

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab_compare, tab_batch, tab_method = st.tabs([
    '📝 Compare Two Snippets',
    '📂 Multi-File Analysis',
    '📚 Methodology',
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Compare Two Snippets
# ════════════════════════════════════════════════════════════════════════════
DEFAULT_A = '''\
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

numbers = [64, 34, 25, 12, 22, 11, 90]
result = bubble_sort(numbers)
print(result)
'''

DEFAULT_B = '''\
def sort_list(data):
    size = len(data)
    for x in range(size):
        for y in range(0, size - x - 1):
            if data[y] > data[y + 1]:
                data[y], data[y + 1] = data[y + 1], data[y]
    return data

my_list = [64, 34, 25, 12, 22, 11, 90]
sorted_result = sort_list(my_list)
print(sorted_result)
'''

with tab_compare:
    st.markdown('Paste two Python snippets and click **Analyze** to measure their semantic similarity.')
    st.caption('💡 The pre-loaded example shows a bubble sort where only variable names differ — a classic obfuscation technique.')

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('#### 📄 Code A  *(original)*')
        code_a = st.text_area('Code A', DEFAULT_A, height=290,
                               label_visibility='collapsed', key='code_a')
        st.caption(f'Lines: {code_a.count(chr(10))+1}  ·  Chars: {len(code_a)}')

    with col_b:
        st.markdown('#### 📄 Code B  *(suspect)*')
        code_b = st.text_area('Code B', DEFAULT_B, height=290,
                               label_visibility='collapsed', key='code_b')
        st.caption(f'Lines: {code_b.count(chr(10))+1}  ·  Chars: {len(code_b)}')

    run_btn = st.button('🔍 Analyze Similarity', type='primary',
                         use_container_width=True, key='run_compare')

    if run_btn:
        if not code_a.strip() or not code_b.strip():
            st.error('Please fill in both code fields.')
        else:
            with st.spinner('Tokenizing, embedding, and computing similarity…'):
                try:
                    # ── Tokenize ──────────────────────────────────────────
                    if normalize:
                        toks_a, map_a = normalize_code(code_a)
                        toks_b, map_b = normalize_code(code_b)
                    else:
                        toks_a = tokenize_python(code_a);  map_a = {}
                        toks_b = tokenize_python(code_b);  map_b = {}

                    stats_a = get_token_stats(toks_a)
                    stats_b = get_token_stats(toks_b)

                    # ── Embed ──────────────────────────────────────────────
                    gv = glove_variant if model_type == 'GloVe' else 'glove-wiki-gigaword-50'
                    (vec_a, vec_b), emb = _embed(
                        [toks_a, toks_b], model_type, w2v_cfg, gv
                    )

                    cov_a = emb.get_token_coverage(toks_a)
                    cov_b = emb.get_token_coverage(toks_b)

                    # ── Similarity ─────────────────────────────────────────
                    score = cosine_similarity(vec_a, vec_b)
                    info  = get_similarity_info(score)

                except Exception as exc:
                    st.error(f'Error during analysis: {exc}')
                    st.exception(exc)
                    st.stop()

            # ── RESULTS ───────────────────────────────────────────────────
            st.markdown('---')
            st.markdown('## 📊 Analysis Results')

            gauge_col, verdict_col = st.columns([1.15, 1])
            with gauge_col:
                st.plotly_chart(make_gauge(score), use_container_width=True)

            with verdict_col:
                st.markdown(f"""
                <div class="verdict-box" style="background:{info['bg_color']};
                     border-color:{info['color']}; color:{info['color']}">
                  <div class="verdict-title">{info['verdict']}</div>
                  <div class="verdict-score">{score:.1%}</div>
                  <div class="verdict-sub">Cosine Similarity Score</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('')
                if score >= threshold:
                    st.error(f'⚠️ Exceeds plagiarism threshold ({threshold:.0%})')
                else:
                    st.success(f'✅ Below plagiarism threshold ({threshold:.0%})')

                st.markdown('')
                st.markdown(f'**Model:** {model_type}  \n'
                             f'**Normalization:** {"On" if normalize else "Off"}')

            # ── Token Stats ────────────────────────────────────────────────
            st.markdown('### 📈 Token Statistics')
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            stats = [
                ('A — Tokens',  stats_a['total_tokens']),
                ('B — Tokens',  stats_b['total_tokens']),
                ('A — Unique',  stats_a['unique_tokens']),
                ('B — Unique',  stats_b['unique_tokens']),
                ('A Coverage',  f'{cov_a:.0%}'),
                ('B Coverage',  f'{cov_b:.0%}'),
            ]
            for col, (lbl, val) in zip([c1, c2, c3, c4, c5, c6], stats):
                col.markdown(
                    f'<div class="stat-card">'
                    f'<div class="stat-label">{lbl}</div>'
                    f'<div class="stat-value">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # ── Token Viewer ───────────────────────────────────────────────
            with st.expander('🔢 Token Output', expanded=False):
                ta, tb = st.columns(2)
                with ta:
                    st.markdown('**Code A**')
                    show_tokens(toks_a)
                with tb:
                    st.markdown('**Code B**')
                    show_tokens(toks_b)

            # ── Identifier Maps ────────────────────────────────────────────
            if normalize:
                with st.expander('🔤 Identifier Normalization Map', expanded=False):
                    ia, ib = st.columns(2)
                    with ia:
                        st.markdown('**Code A**')
                        if map_a:
                            st.dataframe(
                                pd.DataFrame(map_a.items(), columns=['Original', 'Normalized']),
                                hide_index=True, use_container_width=True,
                            )
                        else:
                            st.info('No user-defined identifiers.')
                    with ib:
                        st.markdown('**Code B**')
                        if map_b:
                            st.dataframe(
                                pd.DataFrame(map_b.items(), columns=['Original', 'Normalized']),
                                hide_index=True, use_container_width=True,
                            )
                        else:
                            st.info('No user-defined identifiers.')

            # ── Vector Visualization ───────────────────────────────────────
            with st.expander('📐 Document Vector Comparison', expanded=False):
                st.plotly_chart(make_vector_bar(vec_a, vec_b),
                                use_container_width=True)
                st.caption(
                    'Each bar represents one dimension of the mean-pooled document vector. '
                    'Similar shapes indicate shared semantic structure.'
                )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-File Batch Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown('Upload multiple Python source files to detect plagiarism across an entire submission batch.')

    uploaded = st.file_uploader(
        'Upload .py files (minimum 2)',
        type=['py'],
        accept_multiple_files=True,
    )

    if uploaded:
        if len(uploaded) < 2:
            st.warning('Please upload at least 2 files.')
        else:
            st.success(f'{len(uploaded)} files ready.  Click **Analyze** to proceed.')
            batch_btn = st.button('🔍 Analyze Batch', type='primary',
                                   use_container_width=True, key='run_batch')

            if batch_btn:
                with st.spinner(f'Processing {len(uploaded)} files…'):
                    try:
                        codes = {f.name: f.read().decode('utf-8', errors='replace')
                                 for f in uploaded}

                        all_tokens = {}
                        for name, code in codes.items():
                            if normalize:
                                toks, _ = normalize_code(code)
                            else:
                                toks = tokenize_python(code)
                            all_tokens[name] = toks

                        gv = glove_variant if model_type == 'GloVe' else 'glove-wiki-gigaword-50'
                        vecs_list, emb = _embed(
                            list(all_tokens.values()), model_type, w2v_cfg, gv
                        )

                        labels  = list(all_tokens.keys())
                        sim_df  = pairwise_similarities(vecs_list, labels)
                        suspect = get_suspicious_pairs(sim_df, threshold)

                    except Exception as exc:
                        st.error(f'Error: {exc}')
                        st.exception(exc)
                        st.stop()

                st.markdown('### 🗺️ Similarity Heatmap')
                st.plotly_chart(make_heatmap(sim_df), use_container_width=True)

                st.markdown(f'### 🚨 Flagged Pairs  *(threshold ≥ {threshold:.0%})*')
                if suspect:
                    for f1, f2, sc in suspect:
                        inf = get_similarity_info(sc)
                        css = ('alert-danger' if sc >= 0.95
                               else 'alert-warning' if sc >= 0.80
                               else 'alert-ok')
                        st.markdown(
                            f'<div class="{css}">'
                            f'<b>{f1}</b> ↔ <b>{f2}</b> — '
                            f'<b>{sc:.1%}</b> &nbsp; {inf["verdict"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.success(f'✅ No suspicious pairs found above {threshold:.0%}.')

                with st.expander('📊 Full Similarity Table'):
                    fmt_df = sim_df.copy()
                    fmt_df = fmt_df.map(lambda x: f'{x:.1%}')
                    st.dataframe(fmt_df, use_container_width=True)
    else:
        st.info('👆 Upload 2 or more `.py` files to get started.')


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Methodology
# ════════════════════════════════════════════════════════════════════════════
with tab_method:
    st.markdown("""
## 📚 How It Works

### The Problem with Traditional Checkers
Conventional plagiarism tools compare code **character by character** or use exact token matching.
They fail when students make trivial edits:

| Obfuscation Technique | Traditional Checker | This Tool |
|---|---|---|
| Rename variables (`x` → `count`) | ❌ Miss | ✅ Detect |
| Change comments | ❌ Miss | ✅ Detect |
| Add/remove whitespace | ❌ Miss | ✅ Detect |
| Reorder independent statements | ⚠️ Partial | ⚠️ Partial |
| Change algorithm entirely | ❌ Miss | ❌ Miss |

---

### Pipeline

```
Raw Python Code
      │
      ▼
 ┌─────────────┐
 │ Tokenizer   │  Python `tokenize` module
 │             │  Strips comments, whitespace, strings → semantic tokens
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ Normalizer  │  (optional)
 │             │  var names → var_0, var_1 …
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │  Embedder   │  Word2Vec  OR  GloVe
 │             │  token → 50–200 dimensional vector
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │ Mean Pool   │  Average all token vectors
 │             │  → single document vector
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   Cosine    │  sim(A, B) = (A·B) / (|A| × |B|)
 │  Similarity │  Range: 0.0 (unrelated) → 1.0 (identical)
 └─────────────┘
```

---

### Embedding Models Compared

| Property | Word2Vec | GloVe |
|---|---|---|
| Training data | Your submitted code | Wikipedia + Gigaword (pre-trained) |
| Download size | None (trained on-the-fly) | ~66 MB (cached after first use) |
| Vocabulary | Code-specific tokens | Natural language words |
| Best for | Structural/syntactic patterns | Descriptive identifier names |

---

### Similarity Thresholds

| Score | Verdict |
|---|---|
| ≥ 95% | 🚨 Highly Suspicious — Likely Plagiarized |
| 80–94% | ⚠️ Strongly Suspicious |
| 65–79% | 🔶 Requires Manual Review |
| 40–64% | ✅ Likely Original |
| < 40%  | ✅ Original |

*Thresholds are configurable in the sidebar.*

---

### Limitations & Notes
- **Word2Vec accuracy** improves with larger code corpora. Training on just 2 snippets
  gives approximate results; the tool includes a built-in Python bootstrap corpus to
  stabilize the vocabulary.
- **GloVe** understands English words better than code-specific tokens. Enable
  *Normalize Identifiers* to increase GloVe coverage.
- This tool is a **supporting tool**, not a definitive judge. Always combine automated
  similarity scores with human review.
    """)
