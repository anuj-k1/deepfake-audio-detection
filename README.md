#  Deepfake Audio Detection

A Machine Learning system to classify audio as **Genuine (Human)** or **Deepfake (AI-Generated)**.

##  Results
| Metric | Score |
|--------|-------|
| Accuracy | 96.00% |
| F1 Score | 96.02% |
| EER | 4.25% |

##  Methodology
- **Dataset:** Fake-or-Real Dataset (for-norm subset)
- **Features:** MFCC (40 coefficients) extracted using Librosa
- **Model:** Random Forest Classifier (100 estimators)
- **Train/Test Split:** 80/20

##  Project Structure
