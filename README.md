# ⚗️ DataClean Academy
> **Built for HCL GUVIans** — Interactive data cleaning platform

## ✨ Features
- 6 fully interactive modules with live Flask-powered demos
- Groq llama3-70b AI quiz — module-wise or full syllabus, 5/7/10 questions
- Neelu 🤖 — page-aware AI chatbot tutor, powered by Groq
- White theme · Sora + DM Sans · Teal/Sky/Violet/Pink/Red palette · AOS animations

## 🚀 Local Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY=gsk_your_key_here
python app.py
# → http://localhost:5000
```

## 🌐 Deploy to Render
1. Push to GitHub → New Web Service → connect repo
2. Build: `pip install -r requirements.txt`
3. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
4. Add env var: `GROQ_API_KEY` = your Groq key

## 🔑 Groq API Key
Get a free key at https://console.groq.com → API Keys → Create API Key

## 📁 Structure
```
app.py                  Flask backend + 8 API endpoints
templates/
  base.html             Layout + Neelu chatbot (Groq)
  index.html            Split hero + module grid
  missing_values.html   4-level imputation + live demo
  duplicates.html       CSV table + highlighting
  outliers.html         IQR/Z-score + canvas chart
  skewness.html         Transformations + dot plot
  scaling.html          3 methods + animated bars
  encoding.html         Label/One-Hot/Ordinal + table
  quiz.html             Groq AI quiz + review
```
