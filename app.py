from flask import Flask, render_template, request, jsonify
import json, os, numpy as np, pandas as pd, requests as req_lib
from scipy import stats
from dotenv import load_dotenv

load_dotenv()  # loads .env for local dev; on Render uses env vars directly

app = Flask(__name__)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ── helpers ──────────────────────────────────────────────────────────────────
def safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return None

def clean_list(lst):
    return [safe_float(x) for x in lst if safe_float(x) is not None]

def groq_chat(system_prompt, messages, max_tokens=500):
    """Call Groq API using requests library — same pattern as working RAG app."""
    if not GROQ_API_KEY:
        return "Hey GUVIan! 👋 GROQ_API_KEY not found. Add it to your .env file and restart."
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "temperature": 0.65,
        "max_tokens": max_tokens,
    }
    resp = req_lib.post(GROQ_API_URL, headers=headers, json=payload, timeout=20)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()

# ── pages ────────────────────────────────────────────────────────────────────
@app.route("/ping")
def ping():
    return "ok", 200

@app.route("/")
def index():           return render_template("index.html")

@app.route("/missing-values")
def missing_values():  return render_template("missing_values.html")

@app.route("/duplicates")
def duplicates():      return render_template("duplicates.html")

@app.route("/outliers")
def outliers():        return render_template("outliers.html")

@app.route("/skewness")
def skewness():        return render_template("skewness.html")

@app.route("/scaling")
def scaling():         return render_template("scaling.html")

@app.route("/encoding")
def encoding():        return render_template("encoding.html")

@app.route("/quiz")
def quiz():            return render_template("quiz.html")

# ── API: Neelu chatbot ────────────────────────────────────────────────────────
@app.route("/api/neelu-chat", methods=["POST"])
def api_neelu_chat():
    data       = request.json
    user_msg   = data.get("message", "")
    history    = data.get("history", [])
    page_topic = data.get("page_topic", "data cleaning")
    page_label = data.get("page_label", "DataClean Academy")

    system = f"""You are Neelu 🤖, a friendly and expert AI tutor for DataClean Academy — a data cleaning learning platform built for HCL GUVIans.

Current page: {page_label}
Page topic: {page_topic}

Your personality:
- Always address learners as "GUVIan" (the HCL GUVI community)
- Warm, encouraging, and enthusiastic about data cleaning
- Break down complex concepts simply with real examples
- Use emojis occasionally (not excessively)
- Reference practical pandas/sklearn code when helpful

Your knowledge covers: Missing Value Imputation (MCAR/MAR/MNAR, mean/median/mode, KNN, MICE), Duplicates (drop_duplicates), Outlier Detection (IQR, Z-score), Skewness (log/sqrt/Box-Cox), Feature Scaling (Min-Max, Standardization, Robust), Encoding (Label, One-Hot, Ordinal).

Keep responses concise — 2-5 sentences for simple questions, more for complex ones. Always be helpful, accurate, and never dismissive.
Use **bold** for key terms, _italics_ for emphasis, and `code` for pandas/sklearn snippets."""

    try:
        reply = groq_chat(system, history + [{"role": "user", "content": user_msg}])
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Quiz generation ──────────────────────────────────────────────────────
@app.route("/api/generate-quiz", methods=["POST"])
def api_generate_quiz():
    data   = request.json
    module = data.get("module", "all")
    count  = int(data.get("count", 7))

    module_names = {
        "all":        "Data Cleaning (all topics: missing values, duplicates, outliers, skewness, feature scaling, encoding)",
        "missing":    "Missing Values Imputation (MCAR/MAR/MNAR, mean/median/mode imputation, SimpleImputer, KNNImputer, IterativeImputer)",
        "duplicates": "Duplicates Handling (drop_duplicates, full vs subset, keep parameter, impact on analysis)",
        "outliers":   "Outlier Detection & Handling (IQR, Z-score, winsorization, when to remove vs keep)",
        "skewness":   "Skewness & Transformations (positive/negative skew, log, sqrt, Box-Cox, scipy.stats)",
        "scaling":    "Feature Scaling (Min-Max, Standardization/Z-score, Robust Scaling, when to use each)",
        "encoding":   "Encoding Techniques (Label Encoding, One-Hot Encoding, Ordinal Encoding, dummy variable trap, cardinality)",
    }

    prompt = f"""Generate exactly {count} multiple-choice quiz questions about "{module_names.get(module, module_names['all'])}" for data science learners.

Return ONLY a valid JSON array — no markdown, no preamble, no trailing text:
[
  {{
    "question": "...",
    "category": "Short topic name",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "correct": 0,
    "explanation": "Brief, educational explanation of why this answer is correct."
  }}
]

Rules:
- correct is 0-based index of the correct option in options array
- Mix difficulty: 30% easy, 50% medium, 20% tricky
- Focus on practical, real-world scenarios
- Explanations must be educational and mention the right pandas/sklearn usage when relevant
- Do NOT repeat questions
- Generate exactly {count} questions"""

    system = "You are an expert data science educator generating quiz questions. Return ONLY valid JSON — no markdown fences, no preamble."
    try:
        raw = groq_chat(system, [{"role": "user", "content": prompt}], max_tokens=3000)
        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        start = raw.find("[")
        end   = raw.rfind("]") + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON array found in response")
        questions = json.loads(raw[start:end])
        return jsonify({"questions": questions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ── API: Missing Values ───────────────────────────────────────────────────────
@app.route("/api/impute", methods=["POST"])
def api_impute():
    data   = request.json
    values_raw = data.get("values", [])
    method = data.get("method", "mean")
    values, missing_indices = [], []
    for i, v in enumerate(values_raw):
        if v == "" or v is None:
            missing_indices.append(i); values.append(np.nan)
        else:
            try:    values.append(float(v))
            except: missing_indices.append(i); values.append(np.nan)
    arr   = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return jsonify({"error": "No valid numeric values provided"}), 400
    if method == "mean":     fill = float(np.mean(valid))
    elif method == "median": fill = float(np.median(valid))
    elif method == "mode":   fill = float(stats.mode(valid, keepdims=True).mode[0])
    else:                    fill = float(np.mean(valid))
    result = [{"value": round(fill, 4), "imputed": True} if np.isnan(v) else {"value": round(float(v), 4), "imputed": False} for v in arr]
    return jsonify({
        "result": result, "fill_value": round(fill, 4), "method": method,
        "missing_count": len(missing_indices),
        "stats": {"mean": round(float(np.mean(valid)), 4), "median": round(float(np.median(valid)), 4), "mode": round(float(stats.mode(valid, keepdims=True).mode[0]), 4)}
    })

# ── API: Duplicates ───────────────────────────────────────────────────────────
@app.route("/api/duplicates", methods=["POST"])
def api_duplicates():
    data    = request.json
    records = data.get("records", [])
    subset  = data.get("subset", None)
    if not records:
        return jsonify({"error": "No records provided"}), 400
    df = pd.DataFrame(records)
    original_count = len(df)
    dup_mask = df.duplicated(subset=subset if subset and all(c in df.columns for c in subset) else None, keep="first")
    return jsonify({
        "original_count": original_count,
        "duplicate_count": int(dup_mask.sum()),
        "clean_count": len(df[~dup_mask]),
        "duplicate_indices": df.index[dup_mask].tolist(),
        "clean_records": df[~dup_mask].to_dict(orient="records")
    })

# ── API: Outliers ─────────────────────────────────────────────────────────────
@app.route("/api/outliers", methods=["POST"])
def api_outliers():
    data   = request.json
    values = clean_list(data.get("values", []))
    method = data.get("method", "iqr")
    if len(values) < 3:
        return jsonify({"error": "Need at least 3 numeric values"}), 400
    arr = np.array(values)
    if method == "iqr":
        q1, q3 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (arr < lower) | (arr > upper)
    else:
        z = np.abs(stats.zscore(arr))
        mask = z > 3
        lower, upper = float(arr.mean() - 3*arr.std()), float(arr.mean() + 3*arr.std())
    return jsonify({
        "result": [{"value": round(float(v), 4), "outlier": bool(mask[i])} for i, v in enumerate(arr)],
        "outlier_count": int(mask.sum()), "lower_bound": round(lower, 4), "upper_bound": round(upper, 4),
        "clean_values": [round(float(x), 4) for x in arr[~mask]], "method": method
    })

# ── API: Skewness ─────────────────────────────────────────────────────────────
@app.route("/api/skewness", methods=["POST"])
def api_skewness():
    data      = request.json
    values    = clean_list(data.get("values", []))
    transform = data.get("transform", "log")
    if len(values) < 3:
        return jsonify({"error": "Need at least 3 numeric values"}), 400
    arr = np.array(values)
    original_skew = float(stats.skew(arr))
    if transform == "log":
        shift = abs(arr.min()) + 1 if arr.min() <= 0 else 0
        transformed = np.log(arr + shift)
        note = f"log(x + {round(shift,2)})" if shift else "log(x)"
    elif transform == "sqrt":
        shift = abs(arr.min()) if arr.min() < 0 else 0
        transformed = np.sqrt(arr + shift)
        note = f"sqrt(x + {round(shift,2)})" if shift else "sqrt(x)"
    elif transform == "boxcox":
        shift = abs(arr.min()) + 1 if arr.min() <= 0 else 0
        transformed, lam = stats.boxcox(arr + shift)
        note = f"Box-Cox λ={round(float(lam),4)}" + (f" (shifted +{round(shift,2)})" if shift else "")
    else:
        transformed = arr; note = "None"
    return jsonify({
        "original": [round(float(x), 4) for x in arr],
        "transformed": [round(float(x), 4) for x in transformed],
        "original_skew": round(original_skew, 4),
        "new_skew": round(float(stats.skew(transformed)), 4),
        "transform": transform, "note": note
    })

# ── API: Scaling ──────────────────────────────────────────────────────────────
@app.route("/api/scale", methods=["POST"])
def api_scale():
    data   = request.json
    values = clean_list(data.get("values", []))
    method = data.get("method", "minmax")
    if len(values) < 2:
        return jsonify({"error": "Need at least 2 numeric values"}), 400
    arr = np.array(values)
    if method == "minmax":
        mn, mx = arr.min(), arr.max()
        scaled = (arr - mn) / (mx - mn) if mx != mn else np.zeros_like(arr)
        info = {"min": round(float(mn), 4), "max": round(float(mx), 4)}
    elif method == "standard":
        mu, sigma = arr.mean(), arr.std()
        scaled = (arr - mu) / sigma if sigma != 0 else np.zeros_like(arr)
        info = {"mean": round(float(mu), 4), "std": round(float(sigma), 4)}
    elif method == "robust":
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        med = np.median(arr); iqr = q3 - q1
        scaled = (arr - med) / iqr if iqr != 0 else np.zeros_like(arr)
        info = {"median": round(float(med), 4), "iqr": round(float(iqr), 4)}
    else:
        scaled = arr; info = {}
    return jsonify({"original": [round(float(x), 4) for x in arr], "scaled": [round(float(x), 4) for x in scaled], "method": method, "info": info})

# ── API: Encoding ─────────────────────────────────────────────────────────────
@app.route("/api/encode", methods=["POST"])
def api_encode():
    data   = request.json
    values = data.get("values", [])
    method = data.get("method", "label")
    if not values:
        return jsonify({"error": "No values provided"}), 400
    unique = sorted(set(str(v) for v in values))
    mapping = {v: i for i, v in enumerate(unique)}
    if method in ("label", "ordinal"):
        return jsonify({"method": method, "mapping": mapping, "encoded": [{"original": str(v), "encoded": mapping[str(v)]} for v in values]})
    elif method == "onehot":
        result = [{**{u: (1 if str(v) == u else 0) for u in unique}, "__original__": str(v)} for v in values]
        return jsonify({"method": "onehot", "categories": unique, "encoded": result})
    return jsonify({"error": "Unknown method"}), 400



# ── API: Code Executor ────────────────────────────────────────────────────────
@app.route("/api/run-code", methods=["POST"])
def api_run_code():
    import subprocess as _sp, sys, tempfile, os
    data = request.json
    if not data:
        return jsonify({"output": "", "error": "Invalid request — no JSON body"}), 400
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"output": "", "error": "No code provided"}), 400

    # Safety: block only truly dangerous shell-escape calls
    BLOCKED = ["os.system(", "shutil.rmtree(", "__import__(", "os.popen(", "pty.spawn"]
    for b in BLOCKED:
        if b in code:
            return jsonify({"output": "", "error": f"Blocked: {b} is not allowed"}), 400

    # Preamble — pre-import all libs learners need
    # Pure ASCII preamble - safe on Windows cp1252 and all other encodings
    preamble = (
        "import warnings\n"
        "warnings.filterwarnings('ignore')\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "from scipy import stats\n"
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(preamble + "\n" + code)
        fname = f.name

    try:
        result = _sp.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=12
        )
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        # Strip internal temp-file path from traceback lines
        if stderr:
            cleaned = []
            for line in stderr.splitlines():
                if fname in line:
                    line = line.replace(fname, "your_code.py")
                cleaned.append(line)
            stderr = "\n".join(cleaned)
        return jsonify({"output": stdout, "error": stderr})
    except _sp.TimeoutExpired:
        return jsonify({"output": "", "error": "⏱ Timed out (12s limit) — check for infinite loops"}), 400
    except Exception as e:
        return jsonify({"output": "", "error": str(e)}), 500
    finally:
        try: os.unlink(fname)
        except: pass

# ── API: Personalized feedback message ───────────────────────────────────────
@app.route("/api/personalized-message", methods=["POST"])
def api_personalized_message():
    data         = request.json
    score        = data.get("score", 0)
    total        = data.get("total", 1)
    topic        = data.get("topic", "Data Cleaning")
    wrong_qs     = data.get("wrong_questions", [])
    pct          = round((score / total) * 100) if total else 0

    wrong_summary = ""
    if wrong_qs:
        lines_out = [q["question"] + " (correct: " + q["correct_answer"] + ")" for q in wrong_qs[:5]]
        wrong_summary = "Questions they got wrong: " + "; ".join(lines_out)

    system = "You are Neelu, a warm and encouraging data science learning coach for HCL GUVIans. Give concise, personalised feedback (3-4 sentences max). Always address the learner as 'GUVIan'. Be specific about what to review."
    prompt = f"""A GUVIan just completed a quiz on "{topic}".
Score: {score}/{total} ({pct}%).
{wrong_summary}

Write a short, warm, personalised coaching message. If score < 60%, gently point out which concepts to revisit. If score >= 80%, celebrate and suggest next steps. Be specific and encouraging."""

    try:
        message = groq_chat(system, [{"role": "user", "content": prompt}], max_tokens=200)
        return jsonify({"message": message})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
