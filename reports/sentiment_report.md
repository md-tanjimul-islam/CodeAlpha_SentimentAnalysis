# Sentiment Analysis Report

Reviews analysed: 948 | recommended: 70.0%

## TF-IDF + Logistic Regression: classification report (20% hold-out)
```
                 precision    recall  f1-score   support

Not recommended      0.650     0.709     0.678        55
    Recommended      0.869     0.835     0.851       127

       accuracy                          0.797       182
      macro avg      0.759     0.772     0.765       182
   weighted avg      0.803     0.797     0.799       182

```
## Lexicon vs ML accuracy
| Model | Accuracy | Macro-F1 |
|---|---|---|
| VADER (hold-out, neutral resolved by sign) | 0.786 | 0.726 |
| TF-IDF + Logistic Regression (hold-out) | 0.797 | 0.765 |

VADER on all non-neutral reviews (n=739, 78.0% coverage): accuracy 0.796

Top positive n-grams: fun, good, love, great, best, peak, amaze, nice, awesome, yes, lot, gud, very, addict, recommend
Top negative n-grams: not, no, player, server, fix, dont, garbage, trash, dead, cheater, bad, cant, waste, pve, terrible

## High-playtime dissatisfaction (playtime >= 137.3 h, not recommended)
- 82 reviews = 34.6% of high-playtime reviews and 28.9% of all negative reviews
- Recommendation/VADER mismatches (possible sarcasm or mixed reviews): 151
- Games with most such reviews: HELLDIVERS™ 2 (15); Dota 2 (6); Dead by Daylight (5); Marvel Rivals (5); MapleStory (5)
- Emotion prevalence in this group: Joy/Satisfaction 28%, Anger/Rage 22%, Frustration 15%, Trust/Loyalty 9%, Disappointment 21%
  - "EAC update came out and now game crashes every time i try to launch it. even deleted all my mods and it still crashes."
  - "I'm kinda mixed on this one. Don't get me wrong, the foundation is really solid, artstyle is fun, weapon upgrade system is amazing, but I re"
  - "Sad to see the same problems are still around.. This was one of the best games i had every played as far as combat mechanics, Graphics etc.."

## Emotion prevalence by Steam label (% of reviews)
| label           |   Joy/Satisfaction |   Anger/Rage |   Frustration |   Trust/Loyalty |   Disappointment |
|:----------------|-------------------:|-------------:|--------------:|----------------:|-----------------:|
| Not recommended |               25.7 |         24.6 |          21.5 |             5.6 |             21.5 |
| Recommended     |               47.6 |          4.7 |           6.8 |             3.2 |              4.7 |

## Games with most negative sentiment (>=10 reviews)
| title                      |   n |   mean_compound |   pct_recommended |
|:---------------------------|----:|----------------:|------------------:|
| HELLDIVERS™ 2              |  20 |          -0.293 |                10 |
| STAR WARS: Galactic Racer™ |  20 |          -0.078 |                80 |
| Dead by Daylight           |  20 |          -0.074 |                55 |
| Battlefield™ REDSEC        |  20 |          -0.04  |                30 |
| EA SPORTS FC™ 27           |  20 |           0.004 |                30 |

## Games with most positive sentiment (>=10 reviews)
| title                        |   n |   mean_compound |   pct_recommended |
|:-----------------------------|----:|----------------:|------------------:|
| Disney Dreamlight Valley     |  19 |           0.807 |               100 |
| Dressmaker                   |  20 |           0.573 |               100 |
| Aniimo                       |  20 |           0.498 |                85 |
| Castlevania: Belmont's Curse |  20 |           0.474 |                90 |
| The Elder Scrolls® Online    |  20 |           0.468 |                75 |