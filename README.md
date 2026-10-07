# AI-Powered Medical Diagnosis System

A Python and Streamlit research project that demonstrates machine learning
classification for **diabetes, heart disease, Parkinson's disease, lung cancer,
and hypothyroid conditions** through one interactive web interface.

Select a condition, enter the required measurements, and request a model
prediction. A public dataset example can populate each form to help you explore
how the application works. All five trained models and their preprocessing
pipelines are included, together with reproducible training code and evaluation
reports.

> **Research and education only.** These models are not clinically validated.
> Their predictions are dataset classifications, not medical diagnoses, and a
> negative result cannot rule out disease. Do not use this project to make
> treatment decisions.

## Features

- Five disease classification modules in one Streamlit application.
- Numeric and fractional inputs with required-field and category validation.
- A **Fill public dataset example** button for each module.
- Saved scikit-learn pipelines containing imputation, scaling, and classification.
- Reproducible training with fixed random seeds and documented data sources.
- Evaluation reports with confusion matrices, balanced accuracy, and baseline scores.
- Parkinson's evaluation split by person to avoid sharing a person's recordings
  between the training and test sets.
- An automated check for all five forms, incomplete inputs, and prediction results.

## Technology

Python 3.11, Streamlit, scikit-learn, pandas, NumPy, and Jupyter notebooks.
The heart, diabetes, lung, and thyroid modules use logistic regression.
The Parkinson's module uses a linear support vector classifier.

## Getting started

### Windows PowerShell

```powershell
git clone https://github.com/varunganapuram/medical-diagnosis-using-ai.git
cd medical-diagnosis-using-ai
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501** in your browser. Keep the terminal running while
using the application. This address runs locally; the GitHub repository does not
host the running app.

After installing dependencies, you can also launch the app with `./run.ps1`.
That script binds the server to this computer's loopback interface.

### macOS or Linux

```bash
git clone https://github.com/varunganapuram/medical-diagnosis-using-ai.git
cd medical-diagnosis-using-ai
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The saved models were created with the dependency versions in `requirements.txt`.
Use those versions or retrain the models in your environment. Only load pickle
model files from sources you trust.

## Using the app

1. Select one of the five prediction modules.
2. Choose **Fill public dataset example** to try a test-data example, or enter
   every required measurement yourself.
3. Follow the category codes displayed beside the inputs. Binary fields use
   0 = no/female and 1 = yes/male where their labels indicate that mapping.
4. Press the module's **Test Result** button to view its classification.

Example rows come from public test data. The forms do not save entered values to
a patient database. This project does not implement user authentication, medical
record storage, clinical validation, or compliance certification.

## Evaluation

The following results come from `Models/training_report.json`. They are scores
on small held-out datasets, not guarantees of performance on new patients.

| Module | Test rows | Accuracy | Balanced accuracy | Majority baseline accuracy |
| --- | ---: | ---: | ---: | ---: |
| Diabetes | 154 | 75.97% | 70.43% | 64.94% |
| Heart disease | 54 | 79.63% | 79.58% | 55.56% |
| Parkinson's | 43 | 79.07% | 70.16% | 72.09% |
| Lung cancer | 56 | 92.86% | 80.21% | 85.71% |
| Hypothyroid | 737 | 96.61% | 80.81% | 92.13% |

Non-Parkinson's datasets are deduplicated before a stratified 80/20 split with
seed 2. Parkinson's uses a stratified split of subjects, so the test row count
reflects all recordings belonging to the held-out people. Accuracy can be
misleading when labels are imbalanced; the balanced accuracy and baseline scores
provide additional context.

## Rebuild and verify

The training datasets are included, so no dataset download is needed to retrain.

```powershell
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe verify_app.py
```

On macOS or Linux with the environment activated, use `python train_models.py`
and `python verify_app.py`.

Training regenerates the five `.sav` files, example rows, evaluation report, and
SHA-256 dataset checksum report under `Models/`. The verification script checks
that every page renders, incomplete inputs are rejected, example values populate
correctly, and every prediction button produces a result.

## Project structure

```text
medical-diagnosis-using-ai/
├── app.py                         # Streamlit interface and model loading
├── train_models.py                # Reproducible training for all five modules
├── verify_app.py                  # Streamlit application checks
├── requirements.txt               # Pinned Python dependencies
├── run.ps1                        # Local Windows startup script
├── .streamlit/config.toml         # Theme and usage-stat settings
├── Models/                        # Trained pipelines and evaluation artifacts
├── datasets/                      # Public datasets and source documentation
├── heart_disease_data.csv         # Supplied heart dataset
├── Heart_Disease_Prediction.ipynb  # Original exploratory notebooks
├── Lung_Cancer.ipynb
├── Parkinson's_Disease_Detection.ipynb
└── Thyroid.ipynb
```

Local virtual environments, runtime logs, caches, notebook checkpoints, and
secret files are excluded from Git. The notebooks are retained as references;
their old absolute and Colab paths are superseded by `train_models.py`.

## Data sources and limitations

- **Heart:** the supplied 270-row heart dataset. Its `Presence`/`Absence` labels
  are mapped to 1/0; chest pain, slope, and thallium categories are normalized to
  match the form. It differs from the dataset schema shown in the original notebook.
- **Diabetes:** [Pima dataset mirror maintained by Plotly](https://github.com/plotly/datasets/blob/master/diabetes.csv).
  The dataset represents a specific population and should not be assumed to
  generalize to all people.
- **Parkinson's:** [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/174/parkinsons),
  Little et al.; CC BY 4.0. Uses 22 voice measurements and a split by person.
- **Lung cancer:** [nateraw lung survey dataset](https://huggingface.co/datasets/nateraw/lung-cancer/tree/fdb36446ab5c4d5dbb392517aec7aa755e9a8a51);
  CC BY-NC-SA 4.0. Original symptom codes 1/2 are converted to 0/1. The small,
  imbalanced survey has limited documented provenance; its score is not a
  clinical screening validation.
- **Thyroid:** [UCI Thyroid Disease dataset](https://archive.ics.uci.edu/dataset/102/thyroid+disease),
  Ross Quinlan/Garavan Institute; CC BY 4.0. Uses `allhypo.data` and `allhypo.test`
  with seven selected features. All class names other than `negative`, including
  compensated hypothyroid, map to the positive label.

Training feature order, source references, preprocessing, split sizes, and
metrics are recorded in `Models/training_report.json`. Downloaded file checksums
are recorded in `Models/dataset_checksums.json`. Respect each dataset's own terms;
the lung dataset's license includes a noncommercial restriction.
