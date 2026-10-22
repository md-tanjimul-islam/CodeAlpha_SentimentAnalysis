# CodeAlpha_SentimentAnalysis

**Task 4: Sentiment Analysis | CodeAlpha Data Analytics Internship**

---

## What This Project Does
This project reads Steam user reviews and works out **what players think and feel**. It classifies each review as Positive, Negative or Neutral, scores five gaming-specific emotions, trains a machine-learning model to predict recommendations, and flags long-time players who left negative reviews. The results support marketing and game-development decisions.

## What I Built
One script, `src/sentiment_analysis.py`, with four layers:

1. **Text preprocessing (NLTK):**
   - removes URLs, HTML tags and Steam markup such as `[h1]` and `[b]`;
   - lowercases, expands contractions (`can't` becomes `can not`) and strips special characters;
   - tokenizes, removes stopwords while **keeping negations** like "not" and "never", and lemmatizes.
2. **Lexicon sentiment (VADER):** each review gets a compound score from -1 to +1, labelled Positive (0.05 or above), Negative (-0.05 or below) or Neutral.
3. **Emotion detection:** a hand-built, domain-aware lexicon scores five emotions: Joy/Satisfaction, Anger/Rage, Frustration (Bugs/Optimization), Trust/Loyalty, and Disappointment (Pricing/Content). Simple negation handling means "no bugs" is not counted as frustration.
4. **Machine learning:** a scikit-learn pipeline of `TfidfVectorizer(ngram_range=(1,2), max_features=3000)` plus `LogisticRegression` predicts Steam's recommend flag from the cleaned text, with an 80/20 stratified split.

It also finds **High-Playtime Dissatisfaction**: reviews from players in the top quarter of playtime who did not recommend the game.

## How It Works: Data Flow
```
data/steam_user_reviews.csv
        │
        ├─> preprocessing (clean text)
        │        ├─> TF-IDF + Logistic Regression ─> classification report,
        │        │                                   confusion matrix, top n-grams
        │        └─> emotion lexicon ─> 5 emotion scores
        ├─> VADER on lightly cleaned text ─> polarity label
        └─> playtime + recommendation ─> High-Playtime Dissatisfaction flag
                          │
                          ▼
        data/sentiment_enriched_reviews.csv, plots/, reports/sentiment_report.md
```

VADER reads the markup-cleaned original text because it relies on capitalization and punctuation. The ML model and the emotion lexicon use the normalized text.

## Project Structure
```
CodeAlpha_SentimentAnalysis/
├── data/        # steam_user_reviews.csv in; sentiment_enriched_reviews.csv out
├── plots/       # 4 PNG charts (300 DPI)
├── reports/     # sentiment_report.md (auto-generated)
├── src/sentiment_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## How to Run
```bash
git clone https://github.com/<your-username>/CodeAlpha_SentimentAnalysis.git
cd CodeAlpha_SentimentAnalysis
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Copy `steam_user_reviews.csv` from the Task 1 repo into `data/`, then run:
```bash
python src/sentiment_analysis.py
```
The first run needs internet to download the NLTK data (VADER lexicon, stopwords, WordNet). It does this automatically.

**Outputs**
- `data/sentiment_enriched_reviews.csv`: the original columns plus `clean_text`, `vader_compound`, `vader_label`, five emotion count columns, `dominant_emotion`, `ml_recommend_prob`, `high_playtime`, `high_playtime_dissatisfaction` and `sentiment_mismatch`
- `plots/`: four charts
- `reports/sentiment_report.md`: the classification report, VADER vs ML accuracy table, top n-grams, emotion prevalence, and High-Playtime Dissatisfaction findings from your run

## Visualizations

**1. Sentiment distribution** (VADER labels vs actual Steam recommendations)
![Sentiment Distribution](plots/sentiment_distribution.png)

**2. Emotion profile** (share of reviews expressing each emotion)
![Emotion Profile](plots/emotion_profile_chart.png)

**3. Top TF-IDF keywords** (15 positive and 15 negative n-grams)
![Top Keywords](plots/top_tfidf_keywords.png)

**4. Confusion matrix** (TF-IDF + Logistic Regression)
![Confusion Matrix](plots/confusion_matrix_ml.png)

## Reading the Results
- **Lexicon vs ML:** the report compares VADER and the ML model on the same hold-out reviews. Lexicons need no training and are easy to explain, but they can miss gaming slang and sarcasm. The ML model learns the game vocabulary from the labels, but it depends on how reliable those labels are.
- **Emotions:** the prevalence table shows which emotions are common in recommended and not-recommended reviews.
- **Top n-grams:** the words and phrases that push the model toward a positive or negative prediction.
- **Mismatches:** reviews where the text and the recommend flag disagree, often sarcasm or mixed feelings.

## How to Use the Findings
- **Frustration (Bugs/Optimization) common in negative reviews:** prioritize stability and performance patches and share a fix roadmap.
- **High-Playtime Dissatisfaction reviews:** read them first. Experienced players who turn negative usually point to a specific change or missing end-game content. Use surveys or community updates to respond.
- **Disappointment (Pricing/Content) common:** test discounts or bundles and set clearer expectations on the store page.
- **Trust/Loyalty and Joy common:** reuse the top positive phrases in marketing copy.

Check each of these against the numbers in your own `reports/sentiment_report.md` before relying on it.

## Important Notes
- If the review data came from the synthetic fallback in Task 1, the text is template-generated. The model will score unrealistically well and the emotion results will not reflect real players. Say so in any write-up.
- The emotion lexicon is keyword-based and hand-built, so it can miss sarcasm and unusual phrasing. Edit `EMOTION_LEXICON` in the script to tune it.
- The Steam recommend flag is noisy: players sometimes recommend a game while criticizing it.

---
**Owner:** Md. Tanjimul Islam - CodeAlpha Intern