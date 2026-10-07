"""Train reproducible research models; run with the project virtual environment."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    'diabetes': 'diabetes_model.sav', 'heart_disease': 'heart_disease_model.sav',
    'parkinsons': 'parkinsons_model.sav', 'lung_cancer': 'lungs_disease_model.sav',
    'thyroid': 'Thyroid_model.sav',
}
SOURCES = {
    'diabetes': 'https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv',
    'heart_disease': 'Supplied heart_disease_data.csv and Heart_Disease_Prediction.ipynb',
    'parkinsons': 'https://archive.ics.uci.edu/dataset/174/parkinsons',
    'lung_cancer': 'https://huggingface.co/datasets/nateraw/lung-cancer/tree/fdb36446ab5c4d5dbb392517aec7aa755e9a8a51',
    'thyroid': 'https://archive.ics.uci.edu/dataset/102/thyroid+disease',
}


def datasets():
    diabetes = pd.read_csv(ROOT / 'datasets/diabetes.csv').drop_duplicates()
    yield 'diabetes', diabetes.drop(columns='Outcome'), diabetes.Outcome, None
    heart = pd.read_csv(ROOT / 'heart_disease_data.csv').drop_duplicates()
    if 'Heart Disease' in heart:
        heart.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                         'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        heart['target'] = heart.target.map({'Presence': 1, 'Absence': 0})
        heart['cp'] = heart.cp - 1
        heart['slope'] = heart.slope - 1
        heart['thal'] = heart.thal.map({3: 2, 6: 1, 7: 3})
    yield 'heart_disease', heart.drop(columns='target'), heart.target, None
    parkinsons = pd.read_csv(ROOT / 'datasets/parkinsons/parkinsons.data')
    groups = parkinsons['name'].str.rsplit('_', n=1).str[0]
    yield 'parkinsons', parkinsons.drop(columns=['name', 'status']), parkinsons.status, groups
    lung = pd.read_csv(ROOT / 'datasets/lung_cancer.csv').drop_duplicates()
    lung.columns = lung.columns.str.strip().str.replace(' ', '_')
    lung['GENDER'] = lung.GENDER.map({'M': 1, 'F': 0})
    for column in lung.columns[2:-1]:
        lung[column] = lung[column].map({1: 0, 2: 1})
    yield 'lung_cancer', lung.drop(columns='LUNG_CANCER'), lung.LUNG_CANCER.map({'NO': 0, 'YES': 1}), None
    thyroid = pd.concat([
        pd.read_csv(ROOT / 'datasets/thyroid' / name, header=None, na_values='?')
        for name in ['allhypo.data', 'allhypo.test']
    ], ignore_index=True)
    x = thyroid[[0, 1, 2, 17, 18, 19, 21]].copy()
    x.columns = ['age', 'sex', 'on_thyroxine', 'TSH', 'T3_measured', 'T3', 'TT4']
    x['sex'] = x.sex.map({'M': 1, 'F': 0})
    x['on_thyroxine'] = x.on_thyroxine.map({'f': 0, 't': 1})
    x['T3_measured'] = x.T3_measured.map({'f': 0, 't': 1})
    y = thyroid[29].str.split('.').str[0].ne('negative').astype(int)
    combined = x.assign(target=y).drop_duplicates()
    yield 'thyroid', combined.drop(columns='target'), combined.target, None


def main():
    model_dir = ROOT / 'Models'
    model_dir.mkdir(exist_ok=True)
    report, examples = {}, {}
    for key, x, y, groups in datasets():
        assert not y.isna().any() and set(y.unique()) == {0, 1}
        if groups is None:
            train, test = train_test_split(np.arange(len(x)), test_size=0.2, stratify=y, random_state=2)
        else:
            subjects = pd.DataFrame({'group': groups, 'target': y}).drop_duplicates()
            assert subjects.group.is_unique
            train_groups, test_groups = train_test_split(
                subjects.group, test_size=0.2, stratify=subjects.target, random_state=2
            )
            train = np.flatnonzero(groups.isin(train_groups))
            test = np.flatnonzero(groups.isin(test_groups))
            assert not set(groups.iloc[train]).intersection(groups.iloc[test])
        assert set(y.iloc[train].unique()) == set(y.iloc[test].unique()) == {0, 1}
        estimator = SVC(kernel='linear') if key == 'parkinsons' else LogisticRegression(max_iter=2000)
        model = make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), estimator)
        model.fit(x.iloc[train].to_numpy(dtype=float), y.iloc[train].to_numpy())
        prediction = model.predict(x.iloc[test].to_numpy(dtype=float))
        with (model_dir / MODEL_FILES[key]).open('wb') as output:
            pickle.dump(model, output)
        example = x.iloc[test].dropna().iloc[0]
        examples[key] = [float(value) for value in example]
        majority = int(y.iloc[train].mode().iloc[0])
        report[key] = {
            'source': SOURCES[key], 'features': list(x.columns),
            'rows_after_deduplication': len(x), 'train_rows': len(train), 'test_rows': len(test),
            'test_accuracy': accuracy_score(y.iloc[test], prediction),
            'balanced_accuracy': balanced_accuracy_score(y.iloc[test], prediction),
            'majority_baseline_accuracy': float((y.iloc[test] == majority).mean()),
            'confusion_matrix': confusion_matrix(y.iloc[test], prediction, labels=[0, 1]).tolist(),
            'split': 'subject-disjoint, stratified by subject label, seed=2' if groups is not None else 'stratified, seed=2',
            'positive_label': 'any non-negative allhypo class, including compensated hypothyroid' if key == 'thyroid' else 'dataset disease label=1',
            'sklearn_version': sklearn.__version__,
            'clinical_validation': False,
        }
        print(key, json.dumps(report[key]), flush=True)
    (model_dir / 'training_report.json').write_text(json.dumps(report, indent=2))
    (model_dir / 'examples.json').write_text(json.dumps(examples, indent=2))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (ROOT / 'datasets').rglob('*') if p.is_file()}
    hashes['heart_disease_data.csv'] = hashlib.sha256((ROOT / 'heart_disease_data.csv').read_bytes()).hexdigest()
    (model_dir / 'dataset_checksums.json').write_text(json.dumps(hashes, indent=2))


if __name__ == '__main__':
    main()
