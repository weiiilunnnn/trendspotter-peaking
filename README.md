# TrendSpotter — Streamlit Prototype

Deployment

Interative App Link: https://trendspotter-peaking.streamlit.app/

This app demonstrates the TrendSpotter prototype:
- Micro: token/hashtag growth (last N days vs previous N)
- Momentum: engagement-weighted trend strength
- Segments: approximate audience verticals from topic categories
- Decay Explorer: when a trend starts to fade
- Semantic Topics: Transformer→KMeans (auto TF-IDF fallback)
- Hybrid: micro × macro × momentum (tunable weights)

## 1) Install
```bash
pip install -r requirements.txt
```

> If `sentence-transformers` fails to download a model (no internet), the app will automatically fall back to TF-IDF clustering.

## 2) Run
```bash
streamlit run streamlit_app.py
```

## 3) Data
- Place `videos.csv` in the same folder, **or** use the file uploader in the sidebar.
- Timezone assumed: `Asia/Kuala_Lumpur`.
- Expected columns (missing ones are auto-filled):  
  `videoId, publishedAt, title, description, tags, topicCategories, viewCount, likeCount, commentCount, defaultLanguage, defaultAudioLanguage, contentDuration`

## 4) Technical Stack  

### Languages & Libraries  
- **Python 3.10+**  
- **Pandas, NumPy** — data wrangling  
- **scikit-learn** — clustering, metrics  
- **Sentence-Transformers (MiniLM)** — semantic embeddings  
- **Statsmodels** — trend/decay forecasting  
- **Matplotlib, Streamlit** — visualization & interactive dashboard  

### Deployment  
- Deployed on **Streamlit Community Cloud** for demonstration.  
- Requires no server setup or external cloud costs.  
- Suitable for lightweight prototyping and public sharing.  