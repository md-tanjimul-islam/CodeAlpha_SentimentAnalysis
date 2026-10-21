"""Steam review sentiment: preprocessing, VADER, emotion lexicon, TF-IDF + LR, anomaly detection.

Run from the repo root:  python src/sentiment_analysis.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA, PLOTS, REPORTS = ROOT / "data", ROOT / "plots", ROOT / "reports"
DPI, SEED = 300, 42
REQUIRED = ["review_id", "app_id", "title", "review_text", "recommended",
            "playtime_hours", "helpful_votes", "date_posted"]
sns.set_theme(style="whitegrid", font_scale=1.1)
PAL = {"Positive": "#2a9d8f", "Neutral": "#adb5bd", "Negative": "#d1495b"}
REPORT = []


def say(line=""):
    print(line)
    REPORT.append(str(line))


# ------------------------------------------------------------------ preprocessing
def ensure_nltk():
    for path, pkg in [("sentiment/vader_lexicon.zip", "vader_lexicon"),
                      ("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>|&[a-z]+;|&#\d+;")
BBCODE_RE = re.compile(r"\[/?[a-zA-Z0-9]+(?:=[^\]]*)?\]")  # Steam markup: [h1], [b], [url=..]
CONTRACTIONS = [(r"won't", "will not"), (r"can't", "can not"), (r"n't", " not"), (r"'re", " are"),
                (r"'m", " am"), (r"'ve", " have"), (r"'ll", " will"), (r"'d", " would")]
KEEP = {"no", "nor", "not", "never", "very", "too", "only", "more"}  # sentiment-bearing
DOMAIN_STOP = {"game", "games", "steam"}


def clean_markup(text):
    t = BBCODE_RE.sub(" ", HTML_RE.sub(" ", URL_RE.sub(" ", str(text))))
    return re.sub(r"\s+", " ", t).strip()


def normalize(text):
    t = clean_markup(text).lower().replace("\u2019", "'")
    for pat, rep in CONTRACTIONS:
        t = re.sub(pat, rep, t)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", t)).strip()


def preprocess(df):
    sw = (set(stopwords.words("english")) - KEEP) | DOMAIN_STOP
    lem, tok = WordNetLemmatizer(), RegexpTokenizer(r"[a-z]+")
    df["norm_text"] = df["review_text"].map(normalize)
    df["clean_text"] = df["norm_text"].map(
        lambda t: " ".join(lem.lemmatize(lem.lemmatize(w), "v") for w in tok.tokenize(t)
                           if w not in sw and len(w) > 1))
    return df


# ------------------------------------------------------------------ VADER + emotions
def add_vader(df):
    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["review_text"].map(
        lambda t: float(sia.polarity_scores(clean_markup(t))["compound"]))
    df["vader_label"] = np.select([df["vader_compound"] >= 0.05, df["vader_compound"] <= -0.05],
                                  ["Positive", "Negative"], default="Neutral")
    return df


EMOTION_LEXICON = {  # regex terms matched on normalized text (lowercase, contractions expanded)
    "Joy/Satisfaction": [r"lov(?:e|ed|es|ing)", r"enjoy\w*", r"fun", r"amazing", r"awesome", r"great",
                         r"fantastic", r"excellent", r"masterpiece", r"beautiful", r"gorgeous", r"addict\w*",
                         r"hook(?:ed)?", r"perfect\w*", r"brilliant", r"incredible", r"wonderful", r"best",
                         r"worth it", r"recommend\w*", r"satisf\w*", r"happy", r"relaxing", r"smooth",
                         r"polished", r"can not stop playing", r"10 10", r"hours of fun"],
    "Anger/Rage": [r"hate\w*", r"terrible", r"awful", r"garbage", r"trash", r"scam\w*", r"rip ?off",
                   r"worst", r"disgust\w*", r"furious", r"angry", r"rage\w*", r"stupid", r"joke",
                   r"unacceptable", r"ridiculous", r"cash ?grab", r"greed\w*", r"pathetic", r"insult\w*",
                   r"infuriat\w*", r"annoy\w*", r"sick of", r"fed up", r"toxic"],
    "Frustration (Bugs/Optimization)": [
        r"bugs?", r"buggy", r"crash\w*", r"lag(?:s|gy|ging|ged)?", r"stutter\w*", r"freez\w*", r"froze\w*",
        r"glitch\w*", r"fps", r"frame ?rate\w*", r"frame drops?", r"poor(?:ly)? optimi[sz]\w*",
        r"bad optimi[sz]\w*", r"unoptimi[sz]\w*", r"needs? optimi[sz]\w*", r"broken", r"unplayable",
        r"soft ?lock\w*", r"stuck", r"desync\w*", r"disconnect\w*", r"black screen", r"memory leak",
        r"not (?:launch|load|work|start)\w*", r"performance (?:issues?|problems?)",
        r"loading (?:times?|screens?)", r"hotfix", r"server issues?"],
    "Trust/Loyalty": [r"trust\w*", r"loyal\w*", r"dedicated", r"devs? (?:listen|care|support|communicate|respond)\w*",
                      r"developers? (?:listen|care|support|communicate|respond)\w*", r"regular updates?",
                      r"keep updating", r"constant updates?", r"since (?:launch|beta|release|early access)",
                      r"day (?:one|1)", r"early access", r"community", r"reliable", r"faithful",
                      r"transparent", r"long ?time (?:fan|player)", r"veteran", r"hundreds of hours",
                      r"thousands of hours", r"worth supporting", r"buy again", r"will buy", r"preorder(?:ed)?"],
    "Disappointment (Pricing/Content)": [
        r"disappoint\w*", r"overpric\w*", r"pricey", r"too expensive", r"expensive", r"not worth",
        r"waste of (?:money|time)", r"refund\w*", r"full price", r"price tag", r"dlc", r"micro ?transactions?",
        r"pay to win", r"pay ?wall", r"(?:lack|lacks|lacking) (?:of )?(?:content|depth|variety)",
        r"(?:no|little|not enough|needs? more) content", r"repetitive", r"shallow", r"(?:feels?|is|so) empty",
        r"boring", r"unfinished", r"let down", r"letdown", r"underwhelm\w*", r"bland", r"grind\w*",
        r"not as advertised", r"false advertising", r"too short"],
}
EMOTION_RE = {e: re.compile(r"\b(?:" + "|".join(terms) + r")\b") for e, terms in EMOTION_LEXICON.items()}
EMOTION_COLS = {e: "emo_" + re.sub(r"\W+", "_", e.split(" (")[0].split("/")[0]).lower() for e in EMOTION_LEXICON}
NEGATORS = {"not", "no", "never", "hardly", "without"}


def score_emotions(text):
    out = {}
    for emo, rx in EMOTION_RE.items():
        n = 0
        for m in rx.finditer(text):
            prev = text[max(0, m.start() - 12):m.start()].split()[-1:]
            if not (prev and prev[0] in NEGATORS):  # "no bugs" is not frustration
                n += 1
        out[EMOTION_COLS[emo]] = n
    return out


def add_emotions(df):
    emo = pd.DataFrame(df["norm_text"].map(score_emotions).tolist(), index=df.index)
    df = pd.concat([df, emo], axis=1)
    cols = list(EMOTION_COLS.values())
    inv = {v: k for k, v in EMOTION_COLS.items()}
    df["dominant_emotion"] = np.where(df[cols].max(axis=1) > 0, df[cols].idxmax(axis=1).map(inv), "None")
    return df


# ------------------------------------------------------------------ supervised ML
def build_pipeline():
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000, sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def run_ml(df):
    d = df[df["clean_text"].str.len() > 0]
    y = d["recommended"].astype(int)
    tr, te = train_test_split(d.index, test_size=0.2, random_state=SEED,
                              stratify=y if y.value_counts().min() >= 2 else None)
    pipe = build_pipeline().fit(d.loc[tr, "clean_text"], y[tr])
    pred = pipe.predict(d.loc[te, "clean_text"])
    names = ["Not recommended", "Recommended"]
    report_txt = str(classification_report(y[te], pred, labels=[0, 1], target_names=names,
                                           digits=3, zero_division=0, output_dict=False))
    say("## TF-IDF + Logistic Regression: classification report (20% hold-out)")
    say("```\n" + report_txt + "\n```")

    # out-of-fold probabilities for the enriched CSV (no train/test leakage)
    df["ml_recommend_prob"] = np.nan
    k = min(5, int(y.value_counts().min()))
    if k >= 2:
        oof = cross_val_predict(build_pipeline(), d["clean_text"], y, method="predict_proba",
                                cv=StratifiedKFold(k, shuffle=True, random_state=SEED))[:, 1]
        df.loc[d.index, "ml_recommend_prob"] = oof

    # lexicon vs ML on the same hold-out rows
    vb = (df.loc[te, "vader_compound"] >= 0).astype(int)
    dec = df[df["vader_label"] != "Neutral"]
    v_dec = float((dec["vader_label"].eq("Positive") == dec["recommended"]).mean())
    say("## Lexicon vs ML accuracy")
    say("| Model | Accuracy | Macro-F1 |\n|---|---|---|")
    say(f"| VADER (hold-out, neutral resolved by sign) | {accuracy_score(y[te], vb):.3f} | "
        f"{f1_score(y[te], vb, average='macro'):.3f} |")
    say(f"| TF-IDF + Logistic Regression (hold-out) | {accuracy_score(y[te], pred):.3f} | "
        f"{f1_score(y[te], pred, average='macro'):.3f} |")
    say(f"\nVADER on all non-neutral reviews (n={len(dec)}, {100 * len(dec) / len(df):.1f}% coverage): "
        f"accuracy {v_dec:.3f}")

    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    feats, coef = np.array(vec.get_feature_names_out()), np.asarray(clf.coef_[0], dtype=float)
    order = np.argsort(coef)
    neg = pd.Series(coef[order[:15]], index=feats[order[:15]])
    pos = pd.Series(coef[order[::-1][:15]], index=feats[order[::-1][:15]])
    say("\nTop positive n-grams: " + ", ".join(pos.index))
    say("Top negative n-grams: " + ", ".join(neg.index))
    return confusion_matrix(y[te], pred, labels=[0, 1]), pos, neg


# ------------------------------------------------------------------ anomalies & insights
def high_playtime_dissatisfaction(df):
    thr = max(float(df["playtime_hours"].quantile(0.75)), 1.0)
    df["high_playtime"] = df["playtime_hours"] >= thr
    df["high_playtime_dissatisfaction"] = df["high_playtime"] & ~df["recommended"]
    df["sentiment_mismatch"] = ((df["recommended"] & (df["vader_label"] == "Negative")) |
                                (~df["recommended"] & (df["vader_label"] == "Positive")))
    m = df["high_playtime_dissatisfaction"]
    n_flag, n_high = int(m.sum()), int(df["high_playtime"].sum())
    n_neg = int((~df["recommended"]).sum())
    say(f"\n## High-playtime dissatisfaction (playtime >= {thr:.1f} h, not recommended)")
    say(f"- {n_flag} reviews = {100 * n_flag / max(n_high, 1):.1f}% of high-playtime reviews "
        f"and {100 * n_flag / max(n_neg, 1):.1f}% of all negative reviews")
    say(f"- Recommendation/VADER mismatches (possible sarcasm or mixed reviews): "
        f"{int(df['sentiment_mismatch'].sum())}")
    if n_flag:
        say("- Games with most such reviews: " + "; ".join(
            f"{t} ({n})" for t, n in df[m]["title"].value_counts().head(5).items()))
        prev = (df.loc[m, list(EMOTION_COLS.values())] > 0).astype(float).mean() * 100
        say("- Emotion prevalence in this group: " + ", ".join(
            f"{e.split(' (')[0]} {float(prev[c]):.0f}%" for e, c in EMOTION_COLS.items()))
        for t in df[m].sort_values("helpful_votes", ascending=False)["review_text"].head(3):
            say(f"  - \"{t[:140]}\"")
    return df


def game_and_emotion_insights(df):
    cols = list(EMOTION_COLS.values())
    say("\n## Emotion prevalence by Steam label (% of reviews)")
    flag = df[cols].gt(0).astype(float)
    flag["label"] = df["recommended"].map({True: "Recommended", False: "Not recommended"})
    tab = (flag.groupby("label")[cols].mean() * 100).round(1)
    tab.columns = [e.split(" (")[0] for e in EMOTION_COLS]
    say(tab.to_markdown())
    g = (df.groupby("title").agg(n=("review_id", "count"), mean_compound=("vader_compound", "mean"),
                                 pct_recommended=("recommended", "mean")).query("n >= 10"))
    if len(g):
        g["pct_recommended"] = (g["pct_recommended"].astype(float) * 100).round(1)
        g["mean_compound"] = g["mean_compound"].astype(float).round(3)
        say("\n## Games with most negative sentiment (>=10 reviews)")
        say(g.sort_values("mean_compound").head(5).to_markdown())
        say("\n## Games with most positive sentiment (>=10 reviews)")
        say(g.sort_values("mean_compound", ascending=False).head(5).to_markdown())


# ------------------------------------------------------------------ plots
def save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plots/{name}")


def plot_sentiment_distribution(df):
    order = ["Positive", "Neutral", "Negative"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6))
    counts = df["vader_label"].value_counts().reindex(order).fillna(0)
    sizes = counts.to_numpy(dtype=float)
    a1.pie(sizes, labels=[f"{k}\n{int(v):,}" for k, v in zip(order, sizes)], autopct="%1.1f%%",
           colors=[PAL[k] for k in order], startangle=90, pctdistance=0.8,
           wedgeprops={"width": 0.4, "edgecolor": "white"})
    a1.set_title("VADER classification")
    ct = pd.crosstab(df["recommended"].map({True: "Steam: Recommended", False: "Steam: Not recommended"}),
                     df["vader_label"]).reindex(columns=order, fill_value=0)
    ct.plot(kind="bar", ax=a2, color=[PAL[c] for c in order], rot=0, width=0.8)
    for c in a2.containers:
        a2.bar_label(c, fontsize=9)
    a2.set(xlabel="", ylabel="Reviews", title="VADER label vs actual Steam recommendation")
    a2.legend(title="VADER")
    save(fig, "sentiment_distribution.png")


def plot_emotion_profile(df):
    vals = {e: 100 * float((df[c] > 0).mean()) for e, c in EMOTION_COLS.items()}
    prev = pd.Series(vals).sort_values()
    widths = prev.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(list(prev.index), widths, color=sns.color_palette("mako_r", len(prev)))
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set(xlabel="% of reviews expressing the emotion", title="Emotion Profile of Steam Reviews",
           xlim=(0, max(float(widths.max()) * 1.15, 5.0)))
    save(fig, "emotion_profile_chart.png")


def plot_keywords(pos, neg):
    pos_w = pos.to_numpy(dtype=float)
    neg_w = np.abs(neg.to_numpy(dtype=float))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 7))
    a1.barh(list(pos.index)[::-1], pos_w[::-1], color=PAL["Positive"])
    a1.set_title("Top 15 positive n-grams")
    a2.barh(list(neg.index), neg_w, color=PAL["Negative"])
    a2.invert_yaxis()
    a2.set_title("Top 15 negative n-grams")
    for a in (a1, a2):
        a.set_xlabel("Logistic Regression coefficient (magnitude)")
    fig.suptitle("Terms Driving Player Opinion (TF-IDF + Logistic Regression)", fontsize=15)
    save(fig, "top_tfidf_keywords.png")


def plot_confusion(cm):
    labels = ["Not recommended", "Recommended"]
    pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    annot = np.array([[f"{cm[i, j]}\n({pct[i, j]:.0%})" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", cbar=False, xticklabels=labels,
                yticklabels=labels, ax=ax, annot_kws={"size": 13})
    ax.set(xlabel="Predicted", ylabel="Actual", title="TF-IDF + Logistic Regression: Confusion Matrix")
    save(fig, "confusion_matrix_ml.png")


# ------------------------------------------------------------------ main
def load_reviews():
    path = DATA / "steam_user_reviews.csv"
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}\nPlace steam_user_reviews.csv in data/ and re-run.")
    df = pd.read_csv(path, dtype={"review_id": str})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"steam_user_reviews.csv is missing columns: {missing}")
    df = df.drop_duplicates("review_id").dropna(subset=["review_text", "recommended"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""].copy()
    df["recommended"] = df["recommended"].map(lambda v: str(v).strip().lower() in {"true", "1", "yes"})
    df["playtime_hours"] = pd.to_numeric(df["playtime_hours"], errors="coerce").fillna(0).clip(lower=0)
    df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce").fillna(0).astype(int)
    return df.reset_index(drop=True)


def main():
    ensure_nltk()
    df = load_reviews()
    say(f"# Sentiment Analysis Report\n\nReviews analysed: {len(df)} | "
        f"recommended: {100 * float(df['recommended'].mean()):.1f}%\n")
    df = add_emotions(add_vader(preprocess(df)))
    cm, pos, neg = run_ml(df)
    df = high_playtime_dissatisfaction(df)
    game_and_emotion_insights(df)
    print("\nGenerating plots...")
    plot_sentiment_distribution(df)
    plot_emotion_profile(df)
    plot_keywords(pos, neg)
    plot_confusion(cm)
    out = df.drop(columns=["norm_text"])
    out.to_csv(DATA / "sentiment_enriched_reviews.csv", index=False)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "sentiment_report.md").write_text("\n".join(REPORT), encoding="utf-8")
    print(f"\nSaved data/sentiment_enriched_reviews.csv ({len(out)} rows) and reports/sentiment_report.md")


if __name__ == "__main__":
    main()