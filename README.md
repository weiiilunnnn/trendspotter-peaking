# TrendSpotter — Streamlit Prototype

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

## 4) Notes for Judges
- **No normalization needed** for interpretation; raw scores show natural magnitude differences.
- Hybrid score balances micro (token growth), macro (semantic topic strength), and momentum (engagement). Sliders let you tune weights live.
- If Statsmodels is present, the Decay tab adds a 4-week Holt–Winters forecast.

Enjoy!
