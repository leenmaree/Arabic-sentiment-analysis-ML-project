import re
import csv
import pickle
import pandas as pd
import numpy as np
import nltk
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import ComplementNB
from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def ensure_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


class ArabicPreprocessor:
    NEGATION_WORDS = {
        'لا', 'لم', 'لن', 'ليس', 'ليست', 'ليسوا', 'لسنا', 'لست', 'لستم', 'لستن',
        'ما', 'لما', 'غير', 'بدون', 'دون', 'بلا',
        'مش', 'مو', 'مب', 'ماهو', 'ماهي', 'مهو', 'مهي',
        'ولا', 'أبدا', 'أبداً', 'نهائيا', 'نهائياً',
        'لايمكن', 'لايجوز', 'لاينبغي', 'لايصح',
        'ماكان', 'ماهو', 'مافي', 'مافيش', 'ماعاد', 'ماعندي'
    }

    def __init__(self, use_root3=True, remove_stopwords=True):
        ensure_nltk_resources()
        self.use_root3 = use_root3
        self.remove_stopwords = remove_stopwords
        self.stemmer = ISRIStemmer() if use_root3 else None
        self.stop_words = set(stopwords.words("arabic")) if remove_stopwords else set()

    def clean_tweet(self, tweet):
        tweet = str(tweet).strip()
        tweet = tweet.replace("الله", " ALLAH_TOKEN ")

        tweet = re.sub(r"<.*?>", " ", tweet)
        tweet = re.sub(r"https?://[^\s]+|www\.[^\s]+", " ", tweet)
        tweet = re.sub(r"\brt\b\s*@\S+\s*:\s*", " ", tweet, flags=re.IGNORECASE)
        tweet = re.sub(r"@\S+", " ", tweet)
        tweet = tweet.replace("#", " ").replace("_", " ")

        tweet = re.sub(r"ـ", "", tweet)
        tweet = re.sub(r"[،؛؟!:…\"'()\[\]{}<>]", " ", tweet)
        tweet = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", tweet)
        tweet = re.sub(r"[0-9٠-٩]+", " ", tweet)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U0001FA70-\U0001FAFF"
            "\U00002600-\U000026FF"
            "]",
            flags=re.UNICODE
        )

        tweet = emoji_pattern.sub("", tweet)

        tokens = wordpunct_tokenize(tweet)
        tokens = [t.replace("ALLAH_TOKEN", "الله") for t in tokens]

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words or t in self.NEGATION_WORDS]

        tokens = [t for t in tokens if t.strip()]

        if self.use_root3 and self.stemmer:
            tokens = [self.stemmer.stem(t) if (t != "الله" and t not in self.NEGATION_WORDS) else t
                      for t in tokens]

        return " ".join(tokens).strip()


class FeatureEngineering:
    NEGATION_WORDS = {
        'لا', 'لم', 'لن', 'ليس', 'ليست', 'ليسوا', 'لسنا', 'لست', 'لستم', 'لستن',
        'ما', 'لما', 'غير', 'بدون', 'دون', 'بلا',
        'مش', 'مو', 'مب', 'ماهو', 'ماهي', 'مهو', 'مهي',
        'ولا', 'أبدا', 'أبداً', 'نهائيا', 'نهائياً',
        'لايمكن', 'لايجوز', 'لاينبغي', 'لاyصح',
        'ماكان', 'ماهو', 'مافي', 'مافيش', 'ماعاد', 'ماعندي'
    }

    @staticmethod
    def extract_features(tweet):
        tweet = str(tweet)
        words = tweet.split()

        negation_count = sum(1 for word in words if word in FeatureEngineering.NEGATION_WORDS)

        has_negation = 1 if negation_count > 0 else 0

        negated_words = 0
        for i, word in enumerate(words):
            if word in FeatureEngineering.NEGATION_WORDS:
                negated_words += min(3, len(words) - i - 1)

        return {
            'tweet_length': len(tweet),
            'word_count': len(tweet.split()),
            'exclamation_count': tweet.count('!'),
            'question_count': tweet.count('؟'),
            'dots_count': tweet.count('...') + tweet.count('…'),
            'hashtag_count': tweet.count('#'),
            'mention_count': tweet.count('@'),
            'repeated_chars': len(re.findall(r'(.)\1{2,}', tweet)),
            'has_tatweel': 1 if 'ـ' in tweet else 0,
            'has_negation': has_negation,
            'negation_count': negation_count,
            'negated_context': negated_words,
        }


def read_data():
    base_dir = Path(__file__).resolve().parent
    txt_files = list(base_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt dataset found in: {base_dir}")

    df = pd.read_csv(txt_files[0], sep="\t", header=None, names=["Tweets", "Class"],
                     engine="python", encoding="utf-8", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    df = df.dropna().reset_index(drop=True)
    df["Tweets"] = df["Tweets"].astype(str).str.strip()
    df["Class"] = df["Class"].astype(str).str.strip()
    df = df[(df["Tweets"] != "") & (df["Class"] != "")]
    df["Class"] = df["Class"].replace({"OBJ": "OBJ/NEUTRAL", "NEUTRAL": "OBJ/NEUTRAL"})

    print(f"Loaded dataset: {txt_files[0]}")
    print(f"Total samples: {len(df)}")
    return df, base_dir


def perform_eda(df, save_dir):
    print("\n" + "=" * 50)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 50)

    class_dist = df["Class"].value_counts()
    print(f"\nClass distribution:\n{class_dist}")

    plt.figure(figsize=(8, 6))
    class_dist.plot(kind='bar', color=['#2ecc71', '#e74c3c', '#95a5a6'])
    plt.title('Class Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Sentiment Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_dir / 'class_distribution.png', dpi=300)
    print("Saved: class_distribution.png")
    plt.close()

    return df


def evaluate_model(name, model, X_test, y_test, label_names, save_dir, approach=""):
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{name} {approach}")
    print(f"  Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'{name} {approach}', fontsize=12, fontweight='bold')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    safe_name = name.replace(" ", "_") + approach.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(save_dir / f'cm_{safe_name.lower()}.png', dpi=300)
    plt.close()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def tune_on_validation(model_name, X_train, X_val, y_train, y_val, param_grid):
    best_f1 = 0
    best_params = None
    best_model = None

    print(f"\nTuning {model_name} on validation set...")

    from itertools import product

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    for combo in product(*values):
        params = dict(zip(keys, combo))

        if model_name == "Random Forest":
            m = RandomForestClassifier(**params, random_state=42, class_weight='balanced_subsample')
        elif model_name == "Neural Network":
            m = MLPClassifier(**params, max_iter=200, early_stopping=True, random_state=42)
        elif model_name == "Naive Bayes":
            m = ComplementNB(**params)
        elif model_name == "Decision Tree":
            m = DecisionTreeClassifier(**params, random_state=42, class_weight='balanced')

        m.fit(X_train, y_train)
        y_pred = m.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            best_params = params
            best_model = m

    print(f"Best {model_name} parameters: {best_params}")
    print(f"   Validation F1-Score: {best_f1:.4f}")
    return best_model, best_params


def train_all_models(X_train, X_val, X_test, y_train, y_val, y_test, label_names, save_dir, approach):
    print(f"\n{'=' * 50}")
    print(f"TRAINING MODELS - {approach}")
    print(f"{'=' * 50}")

    all_params = {}

    nb_params = {
        'alpha': [0.01, 0.1, 0.5, 1.0]
    }
    nb, nb_best = tune_on_validation("Naive Bayes", X_train, X_val, y_train, y_val, nb_params)
    all_params['Naive Bayes'] = nb_best

    dt_params = {
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10]
    }
    dt, dt_best = tune_on_validation("Decision Tree", X_train, X_val, y_train, y_val, dt_params)
    all_params['Decision Tree'] = dt_best

    rf_params = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None]
    }
    rf, rf_best = tune_on_validation("Random Forest", X_train, X_val, y_train, y_val, rf_params)
    all_params['Random Forest'] = rf_best

    nn_params = {
        'hidden_layer_sizes': [(64,), (128,), (128, 64)],
        'alpha': [0.0001, 0.001]
    }
    nn, nn_best = tune_on_validation("Neural Network", X_train, X_val, y_train, y_val, nn_params)
    all_params['Neural Network'] = nn_best

    models = {
        'nb': nb,
        'dt': dt,
        'rf': rf,
        'nn': nn
    }

    print(f"\n{'=' * 70}")
    print(f"HYPERPARAMETERS SUMMARY - {approach}")
    print(f"{'=' * 70}")
    for model_name in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'Neural Network']:
        print(f"\n{model_name}:")
        for param, value in all_params[model_name].items():
            print(f"  • {param}: {value}")

    print(f"\n{'=' * 50}")
    print(f"FINAL EVALUATION ON TEST SET - {approach}")
    print(f"{'=' * 50}")

    results = {}
    for name, model_obj in [('Naive Bayes', 'nb'), ('Decision Tree', 'dt'),
                            ('Random Forest', 'rf'), ('Neural Network', 'nn')]:
        results[model_obj] = evaluate_model(name, models[model_obj], X_test,
                                            y_test, label_names, save_dir, approach)

    return models, results, all_params


def interactive_mode(models_imb, models_bal, vectorizer, handcrafted_cols, fe, pre, encoder, label_names):
    print("\n" + "=" * 70)
    print("INTERACTIVE PREDICTION MODE")
    print("=" * 70)

    while True:
        text = input("\nEnter Arabic sentence (or 'quit' to exit): ").strip()
        if text.lower() in ['quit', 'q', 'exit']:
            print("Goodbye!")
            break
        if not text:
            continue

        cleaned = pre.clean_tweet(text)

        handcrafted = pd.DataFrame([fe.extract_features(text)])
        handcrafted = handcrafted[handcrafted_cols]

        tfidf = vectorizer.transform([cleaned])

        X_input = hstack([tfidf, handcrafted.values])

        print(f"\nOriginal: {text}")
        print(f"Cleaned: {cleaned}")
        print("\nIMBALANCED Models:")
        for name, key in [('Naive Bayes', 'nb'), ('Decision Tree', 'dt'),
                          ('Random Forest', 'rf'), ('Neural Network', 'nn')]:
            pred = label_names[models_imb[key].predict(X_input)[0]]
            print(f"  {name:<20} -> {pred}")

        print("\nBALANCED Models (SMOTE):")
        for name, key in [('Naive Bayes', 'nb'), ('Decision Tree', 'dt'),
                          ('Random Forest', 'rf'), ('Neural Network', 'nn')]:
            pred = label_names[models_bal[key].predict(X_input)[0]]
            print(f"  {name:<20} -> {pred}")


if __name__ == "__main__":
    print("=" * 70)
    print("ARABIC SENTIMENT ANALYSIS PROJECT")
    print("=" * 70)

    df, base_dir = read_data()
    df = perform_eda(df, base_dir)

    print("\n" + "=" * 50)
    print("PREPROCESSING")
    print("=" * 50)
    pre = ArabicPreprocessor()
    df["CleanTweets"] = df["Tweets"].apply(pre.clean_tweet)
    print("Cleaned all tweets")

    print("\n" + "=" * 50)
    print("FEATURE EXTRACTION")
    print("=" * 50)
    fe = FeatureEngineering()
    handcrafted = pd.DataFrame(df["Tweets"].apply(fe.extract_features).tolist())
    print(f"Extracted {len(handcrafted.columns)} hand-crafted features")

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Class"])
    label_names = list(encoder.classes_)

    print("\n" + "=" * 50)
    print("DATA SPLIT")
    print("=" * 50)

    train_idx, temp_idx = train_test_split(
        np.arange(len(df)), test_size=0.4, random_state=42, stratify=y
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx]
    )

    train_tweets = df["CleanTweets"].iloc[train_idx].reset_index(drop=True)
    val_tweets = df["CleanTweets"].iloc[val_idx].reset_index(drop=True)
    test_tweets = df["CleanTweets"].iloc[test_idx].reset_index(drop=True)

    handcrafted_train = handcrafted.iloc[train_idx].reset_index(drop=True)
    handcrafted_val = handcrafted.iloc[val_idx].reset_index(drop=True)
    handcrafted_test = handcrafted.iloc[test_idx].reset_index(drop=True)

    y_train = y[train_idx]
    y_val = y[val_idx]
    y_test = y[test_idx]

    print(f"Train: {len(y_train)} | Validation: {len(y_val)} | Test: {len(y_test)}")

    unique, counts = np.unique(y_train, return_counts=True)
    print("\nOriginal class distribution in training:")
    for label_idx, count in zip(unique, counts):
        print(f"  {label_names[label_idx]}: {count}")

    print("\nCRITICAL: Fitting TF-IDF ONLY on training data (no leakage)...")
    vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"[^\s]+", ngram_range=(1, 2))
    tfidf_train = vectorizer.fit_transform(train_tweets)
    tfidf_val = vectorizer.transform(val_tweets)
    tfidf_test = vectorizer.transform(test_tweets)
    print(f"TF-IDF features: {tfidf_train.shape[1]} (fitted on training only)")

    with open("vectorizer.pickle", "wb") as f:
        pickle.dump(vectorizer, f)

    X_train = hstack([tfidf_train, handcrafted_train.values])
    X_val = hstack([tfidf_val, handcrafted_val.values])
    X_test = hstack([tfidf_test, handcrafted_test.values])
    print(f"Total features: {X_train.shape[1]}")

    print("\n" + "=" * 70)
    print("APPROACH 1: IMBALANCED DATA (Original)")
    print("=" * 70)
    models_imb, results_imb, params_imb = train_all_models(X_train, X_val, X_test, y_train, y_val, y_test,
                                                           label_names, base_dir, "(IMBALANCED)")

    print("\n" + "=" * 70)
    print("APPROACH 2: BALANCED DATA (SMOTE)")
    print("=" * 70)
    print("Applying SMOTE to training data only...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    unique_bal, counts_bal = np.unique(y_train_bal, return_counts=True)
    print(f"Training size: {len(y_train)} -> {len(y_train_bal)}")
    print("\nBalanced class distribution:")
    for label_idx, count in zip(unique_bal, counts_bal):
        print(f"  {label_names[label_idx]}: {count}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.bar(label_names, counts, color=['#e74c3c', '#95a5a6', '#2ecc71'])
    ax1.set_title('BEFORE SMOTE', fontweight='bold')
    ax1.set_ylabel('Count')
    for i, v in enumerate(counts):
        ax1.text(i, v, str(v), ha='center', va='bottom')

    ax2.bar(label_names, counts_bal, color=['#e74c3c', '#95a5a6', '#2ecc71'])
    ax2.set_title('AFTER SMOTE', fontweight='bold')
    ax2.set_ylabel('Count')
    for i, v in enumerate(counts_bal):
        ax2.text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(base_dir / 'smote_comparison.png', dpi=300)
    print("Saved: smote_comparison.png")
    plt.close()

    models_bal, results_bal, params_bal = train_all_models(X_train_bal, X_val, X_test, y_train_bal, y_val, y_test,
                                                           label_names, base_dir, "(BALANCED)")

    print("\n" + "=" * 70)
    print("COMPARISON: IMBALANCED vs BALANCED")
    print("=" * 70)

    model_keys = ['nb', 'dt', 'rf', 'nn']
    model_names_full = ['Naive Bayes', 'Decision Tree', 'Random Forest', 'Neural Network']

    for name, key in zip(model_names_full, model_keys):
        print(f"\n{name}:")
        print(f"  Metric      | IMBALANCED | BALANCED   | Improvement")
        print(f"  {'-' * 55}")
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            imb = results_imb[key][metric]
            bal = results_bal[key][metric]
            diff = bal - imb
            print(f"  {metric.capitalize():<11} | {imb:.4f}     | {bal:.4f}     | {diff:+.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('IMBALANCED vs BALANCED Comparison', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(model_keys))
        width = 0.35

        imb_vals = [results_imb[k][metric] for k in model_keys]
        bal_vals = [results_bal[k][metric] for k in model_keys]

        ax.bar(x - width / 2, imb_vals, width, label='Imbalanced', color='#e74c3c')
        ax.bar(x + width / 2, bal_vals, width, label='Balanced', color='#2ecc71')
        ax.set_title(metric.upper(), fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_xticks(x)
        ax.set_xticklabels(['NB', 'DT', 'RF', 'NN'])
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(base_dir / 'comparison.png', dpi=300)
    print("\nSaved: comparison.png")
    plt.close()

    print("\n" + "=" * 50)
    print("SAVING MODELS (Balanced versions)")
    print("=" * 50)
    for name, model in models_bal.items():
        with open(f"{name}_model.pickle", "wb") as f:
            pickle.dump(model, f)
    with open("label_encoder.pickle", "wb") as f:
        pickle.dump(encoder, f)
    print("All models saved!")

    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print(f"Generated 8 confusion matrices")
    print(f"Generated 3 plots (EDA + SMOTE + Comparison)")
    print(f"Files saved in: {base_dir}")
    print("\nFIXED: TF-IDF fitted ONLY on training data (no leakage)")
    print("FIXED: Validation set used properly for hyperparameter tuning")

    print("\n" + "=" * 70)
    print("FINAL HYPERPARAMETERS REPORT")
    print("=" * 70)
    print("\nIMBALANCED Approach:")
    for model_name in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'Neural Network']:
        print(f"\n{model_name}:")
        for param, value in params_imb[model_name].items():
            print(f"  • {param}: {value}")

    print("\n" + "-" * 70)
    print("BALANCED Approach (with SMOTE):")
    for model_name in ['Naive Bayes', 'Decision Tree', 'Random Forest', 'Neural Network']:
        print(f"\n{model_name}:")
        for param, value in params_bal[model_name].items():
            print(f"  • {param}: {value}")

    print("\n" + "=" * 70)
    user_choice = input("Test your own Arabic sentences? (yes/no): ").strip().lower()
    if user_choice in ['yes', 'y']:
        interactive_mode(models_imb, models_bal, vectorizer, handcrafted_train.columns,
                         fe, pre, encoder, label_names)
    else:
        print("Thank you! Program ended.")# your code goes here