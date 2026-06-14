#  Deepfake Audio Detection

A Machine Learning system to classify audio as Genuine (Human) or Deepfake (AI-Generated).

##  Results
| Metric | Score |
|--------|-------|
| Accuracy | 96.00% |
| F1 Score | 96.02% |
| EER | 4.25% |

##  Methodology
- Dataset: Fake-or-Real Dataset (for-norm subset)
- Features: MFCC (40 coefficients) extracted using Librosa
- Model: Random Forest Classifier (100 estimators)
- Train/Test Split: 80/20

##  Project Structure
deepfake-audio-detection/
├── app.py              
├── deepfake_model.pkl  
├── notebook.ipynb      
└── README.md           

##  How to Run
pip install streamlit librosa scikit-learn numpy
streamlit run app.py

##  Web App
Upload any .wav or .mp3 file to detect if it is Genuine or Deepfake with confidence score.

##  Confusion Matrix
[[191   8]
 [  8 193]]

##  Pipeline
1. Audio loaded and resampled to 16kHz
2. MFCC features extracted (40 coefficients)
3. Mean taken across time axis
4. Random Forest classifies as Real or Fake
