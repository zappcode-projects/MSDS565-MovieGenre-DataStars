# Movie Genre Classification from Plot Synopses

MSDS 565, Project 3 (Unit 3, text classification). We explored a corpus of movie plot synopses, trained one logistic regression per genre on TF-IDF and on sentence embeddings, measured how much accuracy is lost when an LLM supplies the training labels instead of IMDB, and explained the TF-IDF models with coefficients and SHAP.

**Team:** DataStars &middot; Michael Flanders, Justin Irick, Antonio Brown

## 1. Data

All three notebooks read one file, `mpst_imdb_movies.csv` (about 11,800 movies, one row per movie), provided by the course. It was built by joining two public datasets on the IMDB ID:

- **MPST: Movie Plot Synopses with Tags** supplies the plot synopses (from Wikipedia or IMDB) and the train/val/test split. https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags
- **IMDB Non-Commercial Datasets** supply the genres and the metadata (start year, runtime, rating, votes). https://developer.imdb.com/non-commercial-datasets/

**No data is in this repository.** Put the provided CSV at `~/MSDS565_data/mpst_imdb_movies.csv` (on Windows, `C:\Users\<you>\MSDS565_data\`). Notebook 2 writes its cache (embedding matrix, teacher labels, `pseudo_labeled.csv`) to `~/MSDS565_data/P3_cache/`, also outside the repo.

The only model input is `plot_synopsis`. The IMDB columns are used for the vote filter and for analysis only.

## 2. What each notebook produces, in order

Run the notebooks in order, from the `notebooks/` folder. Each reads what the one before it saved.

| # | Notebook | What it does | Produces |
|---|---|---|---|
| 1 | `P3_MovieGenre-1_EDA_DataStars.ipynb` | Explores the movie characteristics, the genre balance for all genres and the synopsis text; chooses the vote filter | Nothing on disk. Its decisions (vote filter `numVotes >= 500`, credit-line cleaning) are applied in Notebooks 2 and 3 |
| 2 | `P3_MovieGenre-2_Modeling_DataStars.ipynb` | Selects 3 genres, builds TF-IDF and sentence embeddings, standardizes them, trains class-weighted logistic regressions, runs LLM teacher annotation and trains students on the teacher's labels, ranks everything on a leaderboard | `models/P3_N2_<rep>_logreg_<genre>.pkl` (six files), `models/P3_N2_tfidf_vectorizer.pkl`, `models/P3_N2_tfidf_scaler.pkl`; cache files outside the repo |
| 3 | `P3_MovieGenre-3_xAI_DataStars.ipynb` | Explains the three TF-IDF models: top coefficients, SHAP bar and beeswarm plots, and waterfall plots for individual movies | Nothing on disk (analysis only) |

## 3. Models and the teacher LLM

- **Selected genres:** Comedy, Horror, Romance (each has at least 5% positives in both the training and test sets; roughly 30%, 18% and 18%).
- **Representations:** TF-IDF (unigrams, English stop words, `min_df=5`, `max_df=0.5`, 20,000 features, sublinear TF) and one sentence-embedding model. Both are standardized with `StandardScaler` (`with_mean=False` for TF-IDF), fit on the training set only.
- **Embedding model ID:** `sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions. Synopses are longer than the model's input limit, so each is split into 150-word chunks, the chunk embeddings are averaged and re-normalized. https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- **Classifier:** logistic regression for every model. `C` is fixed once per representation by cross-validation; `class_weight` is tuned per genre with `GridSearchCV` (6 settings, 5-fold stratified, macro-F1).
- **Teacher LLM:** Claude Haiku 5.5 (`claude-haiku-5-5`) through the Anthropic API, with a JSON-schema constrained reply (one required true/false field per genre). The teacher labels at most 300 training plots (100 per genre, `random_state=42`), never sees titles or the test set, and its labels are cached so a rerun costs nothing.
- **Ollama:** not used. There is nothing to pull. Notebook 2 needs an Anthropic API key in the `ANTHROPIC_API_KEY` environment variable (see section 5). Without a cached label file the teacher step makes about 300 short API calls.

## 4. Results

Test-set macro-F1, one classifier per genre (Notebook 2 leaderboard):

| Model (training labels) | Comedy | Horror | Romance | Mean |
|---|---|---|---|---|
| TF-IDF (IMDB) | 0.789 | 0.876 | 0.743 | 0.803 |
| Embedding (IMDB) | 0.747 | 0.834 | 0.716 | 0.766 |
| TF-IDF (LLM labels) | 0.621 | 0.656 | 0.631 | 0.636 |
| Embedding (LLM labels) | 0.618 | 0.634 | 0.576 | 0.610 |
| Teacher on its sample (reference, not on the test set) | 0.852 | 0.925 | 0.802 | 0.860 |

- TF-IDF beats the embedding on every genre. Horror is the easiest genre and Romance the hardest.
- Training on 261 LLM-labelled plots instead of about 8,900 IMDB-labelled ones costs about 0.16 macro-F1 on average. 39 of the 300 teacher replies came back refused or cut off, so 261 plots were used. Reading the teacher's disagreements with IMDB shows that much of the gap is differences between IMDB's broad genre tags and the stricter definition in our prompt.
- Notebook 3 shows that the largest coefficients belong to rare words (for example `vampire` for Horror), while the largest SHAP values belong to common words (`body`, `blood`, `dead`, `love`). The SHAP ranking better describes what drives predictions on real movies.

## 5. How to reproduce

1. Create the environment: `conda env create -f environment.yml`, then `conda activate msds565`. (Rename the environment in the first line of the file if you want a different name.)
2. Put `mpst_imdb_movies.csv` in `~/MSDS565_data/`.
3. For Notebook 2 only, set the key before starting Jupyter or VS Code. On Windows: `setx ANTHROPIC_API_KEY your-key`, then restart the editor. On macOS or Linux: `export ANTHROPIC_API_KEY=your-key`. Never put the key in a notebook or a file in this repo.
4. Open the notebooks from the `notebooks/` folder and run them in order, top to bottom. Notebook 2 takes several minutes the first time (embedding the synopses and the 300 teacher calls) and seconds after that, because both are cached.

## 6. Repository layout

```
MSDS565-MovieGenre-DataStars/
├── README.md
├── environment.yml
├── models/            # written by Notebook 2, read by Notebook 3
│   ├── P3_N2_tfidf_logreg_<genre>.pkl
│   ├── P3_N2_emb-minilm_logreg_<genre>.pkl
│   ├── P3_N2_tfidf_vectorizer.pkl
│   └── P3_N2_tfidf_scaler.pkl
└── notebooks/
    ├── P3_MovieGenre-1_EDA_DataStars.ipynb
    ├── P3_MovieGenre-2_Modeling_DataStars.ipynb
    └── P3_MovieGenre-3_xAI_DataStars.ipynb
```
