import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.metrics import make_scorer, mean_absolute_error, accuracy_score
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

print("Inizio del programma")
chunks = pd.read_csv("bilotta/spotibatch.csv", chunksize=100_000)

df = pd.concat(
    [chunk.dropna() for chunk in chunks],
    ignore_index=True
).drop_duplicates()

df = df[(df['popularity'] > 2) & (df['danceability'] > 0) & (df['energy'] > 0)]
df = df.sort_values(by='popularity', ascending=False)

print(f"Dataset shape: {df.shape}")
print(df.head())
print("Dataframe caricato e pre-elaborato")

soglia = df['popularity'].quantile(0.7)  # Top 30% come popolari
print(f"Usando soglia di popularity: {soglia}")

df['popular'] = (df['popularity'] > soglia).astype(int)
print("\nDistribuzione classi:")
print(df['popular'].value_counts(normalize=True))

macro_genre = {
    'Rock & Metal': [
        'rock', 'alt-rock', 'metal', 'black-metal', 'punk', 'punk-rock',
        'metalcore', 'hardcore', 'heavy-metal', 'hard-rock', 'psych-rock',
        'grindcore', 'emo', 'death-metal', 'rock-n-roll', 'indie-rock'
    ],
    'Electronic': [
        'electronic', 'techno', 'trance', 'house', 'deep-house', 'edm', 'electro',
        'ambient', 'drum-and-bass', 'dubstep', 'chicago-house', 'detroit-techno',
        'progressive-house', 'minimal-techno', 'breakbeat', 'dancehall', 'club'
    ],
    'Pop': [
        'pop', 'alt-pop', 'pop-film', 'indie-pop', 'k-pop', 'dance', 'romance',
        'power-pop', 'party', 'cantopop', 'j-pop', 'synthpop'
    ],
    'Folk & Acoustic': [
        'folk', 'acoustic', 'singer-songwriter', 'songwriter', 'country',
        'forro', 'bluegrass', 'americana'
    ],
    'Jazz & Soul': [
        'jazz', 'blues', 'funk', 'soul', 'groove', 'bossanova', 'r-n-b', 'gospel'
    ],
    'World': [
        'spanish', 'french', 'german', 'swedish', 'tango', 'samba', 'salsa',
        'sertanejo', 'afrobeat', 'garage', 'indian', 'reggae', 'reggaeton', 'latin'
    ],
    'Classical & Instrumental': [
        'classical', 'piano', 'opera', 'show-tunes', 'new-age', 'guitar', 'orchestral'
    ],
    'Hip-Hop & Urban': [
        'hip-hop', 'trip-hop', 'dub', 'trap', 'grime', 'rap'
    ]
}

def trova_macro_genere(genere):
    """Trova il macro genere per un genere specifico"""
    genere_lower = genere.lower() if pd.notna(genere) else ''
    for macro, sottogeneri in macro_genre.items():
        if any(sub in genere_lower for sub in sottogeneri):
            return macro
    return 'Altri'

df['macro_genre'] = df['genre'].apply(trova_macro_genere)
print("\nDistribuzione macro generi:")
print(df['macro_genre'].value_counts())

feature_cols = [
    'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms',
    'key', 'time_signature', 'macro_genre', 'mode', 'year'
]

X = df[feature_cols]
y = df['popular']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numerical_features = ['danceability', 'energy', 'loudness', 'speechiness',
                     'acousticness', 'instrumentalness', 'liveness', 'valence',
                     'tempo', 'duration_ms']

categorical_features = ['key', 'time_signature', 'macro_genre']

ordinal_features = ['mode', 'year']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('ord', OrdinalEncoder(handle_unknown='ignore'), ordinal_features)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        random_state=42,
        eval_metric='logloss'
    ))
])

params = {
    'classifier__n_estimators': randint(100, 500),
    'classifier__max_depth': randint(3, 10),
    'classifier__learning_rate': uniform(0.01, 0.3)
}

random_search = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=params,
    n_iter=5,
    scoring=make_scorer(mean_absolute_error, greater_is_better=False),
    # cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    # n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train, y_train)

print("\nMigliori parametri trovati:")
for param, value in random_search.best_params_.items():
    print(f"{param}: {value}")

best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred)
ACC = accuracy_score(y_test, y_pred)

print(f'''
Mean Absolute Error = {MAE}
Accuracy = {ACC}
''')