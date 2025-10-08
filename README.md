# ⚾ MLB Wins Prediction with Machine Learning  

## Overview  
This project explores how team statistics can be transformed into predictive features to estimate **Major League Baseball (MLB) team wins**.  

Using historical data, the notebook applies feature engineering, lag statistics, and regression models to uncover the drivers of team performance.  

💡 This project came out **top in an internal Kaggle-style competition**, achieving the **lowest Mean Absolute Error (MAE)** among all submissions.

## Key Original Features in the dataset

| Feature       | Category         | Description                                 |
| ------------- | ---------------- | ------------------------------------------- |
| `G`           | Basic            | Games played                                |
| `R`           | Batting          | Runs scored                                 |
| `AB`          | Batting          | At-bats                                     |
| `H`           | Batting          | Hits                                        |
| `2B`          | Batting          | Doubles                                     |
| `3B`          | Batting          | Triples                                     |
| `HR`          | Batting          | Home runs                                   |
| `BB`          | Batting          | Walks                                       |
| `SO`          | Batting          | Strikeouts                                  |
| `SB`          | Batting          | Stolen bases                                |
| `RA`          | Pitching/Defense | Runs allowed                                |
| `ER`          | Pitching/Defense | Earned runs allowed                         |
| `ERA`         | Pitching/Defense | Earned run average                          |
| `CG`          | Pitching/Defense | Complete games                              |
| `SHO`         | Pitching/Defense | Shutouts                                    |
| `SV`          | Pitching/Defense | Saves                                       |
| `IPouts`      | Pitching/Defense | Outs pitched (innings × 3)                  |
| `HA`          | Pitching/Defense | Hits allowed                                |
| `HRA`         | Pitching/Defense | Home runs allowed                           |
| `BBA`         | Pitching/Defense | Walks allowed                               |
| `SOA`         | Pitching/Defense | Pitcher strikeouts                          |
| `E`           | Pitching/Defense | Errors                                      |
| `DP`          | Pitching/Defense | Double plays                                |
| `FP`          | Pitching/Defense | Fielding percentage                         |
| `mlb_rpg`     | Derived          | MLB average runs per game (season-specific) |
| `era_1`       | Era Indicator    | Pre-1920: Dead-ball era                     |
| `era_2`       | Era Indicator    | 1920–1941: Live-ball era                    |
| `era_3`       | Era Indicator    | 1942–1945: WWII era                         |
| `era_4`       | Era Indicator    | 1946–1962: Post-war era                     |
| `era_5`       | Era Indicator    | 1963–1976: Pitcher's era                    |
| `era_6`       | Era Indicator    | 1977–1992: Free agency era                  |
| `era_7`       | Era Indicator    | 1993–2009: Steroid era                      |
| `era_8`       | Era Indicator    | 2010–present: Post-steroid/analytics era    |
| `decade_1910` | Decade Indicator | 1910s                                       |
| `decade_1920` | Decade Indicator | 1920s                                       |
| `decade_1930` | Decade Indicator | 1930s                                       |
| `decade_1940` | Decade Indicator | 1940s                                       |
| `decade_1950` | Decade Indicator | 1950s                                       |
| `decade_1960` | Decade Indicator | 1960s                                       |
| `decade_1970` | Decade Indicator | 1970s                                       |
| `decade_1980` | Decade Indicator | 1980s                                       |
| `decade_1990` | Decade Indicator | 1990s                                       |
| `decade_2000` | Decade Indicator | 2000s                                       |
| `decade_2010` | Decade Indicator | 2010s                                       |


## 🔑 Key Steps  
- **Feature Engineering**: Created per-game metrics (e.g. runs/game, ERA, OBP) and lag features from prior seasons to strengthen predictive signals.  
- **Model Training**: Applied **Kernel Ridge Regression (KRR)**, which balances Ridge Regression’s regularisation with kernel methods to capture non-linear relationships (e.g. between runs, OBP, and pitching). This avoided the need for extra polynomial or interaction features.  
- **Optimisation**: Tuned hyperparameters with `GridSearchCV` to achieve the best trade-off between accuracy and generalisation.  
- **Evaluation**: Assessed model performance using Mean Absolute Error (MAE).  
- **Workflow**: Designed a clean pipeline from raw stats → engineered features → model training → predictions.  

## 📊 Insights  
- Per-game statistics (like runs scored or allowed) provide stronger predictive signals than raw totals.  
- **Kernel Ridge Regression outperformed XGBoost and other more complex models**, which in fact produced worse MAE.  

## 📝 Key Takeaways 
1. **Start Simple**: Simpler models can sometimes deliver better results.  
2. **Feature Engineering Matters Most**: The single biggest factor in reducing MAE was careful feature engineering, followed by **choosing the right model**, and only then **hyperparameter tuning**.  
3. **Per-Game Normalisation Beats Raw Totals**: Normalised features (e.g. runs per game) improved predictive power.  
4. **Regularisation Prevents Overfitting**: L2 regularisation in KRR helped stabilise results.  
5. **Interpretability Counts**: A simpler, well-regularised model gave clearer insights and more reliable outcomes than a “black box” approach.  

---

⚡ *This project shows how data science can help decode the game of baseball — turning stats into wins predictions.*  

More information can be found here: https://www.kaggle.com/competitions/sctpdsai-module-3-coaching-3-1-money-ball/data
