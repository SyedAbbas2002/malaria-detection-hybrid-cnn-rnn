# Advancing Malaria Identification From Microscopic Blood Smears Using Hybrid Deep Learning Frameworks

This project uses **Hybrid Deep Learning Models** (CNN + RNNs) to classify malaria-infected blood smears from microscopic images.

## Project Structure
- `src/` — Python source code (data loading, model building, training, evaluation, prediction)
- `dataset/` — Folder to place your malaria dataset
- `models/` — Saved trained models
- `utils/` — Utility scripts
- `.gitignore` — Files/folders to ignore when uploading to GitHub
- `requirements.txt` — Python libraries list

## Deep Learning Architecture
- **Feature Extraction**: 2D CNN
- **Classifier**: Two cascaded RNN layers (GRU, LSTM, Bi-LSTM combinations)

## Dataset Download Instructions
This project uses the [Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) available on Kaggle.

To set up:

1. Download the dataset ZIP from [Kaggle Link](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria).
2. Extract it.
3. Move the extracted folders (`Parasitized/` and `Uninfected/`) inside the `dataset/` folder.

Folder structure after moving:

```
dataset/
├── Parasitized/
│   ├── C100P61ThinF_IMG_20150918_144104_cell_162.png
│   └── ...
├── Uninfected/
│   ├── C100P61ThinF_IMG_20150918_144104_cell_163.png
│   └── ...
```

**Important:** 
- Images will automatically be loaded during training (`train.py`)!

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
python src/predict.py
```

## Results
- Best Model: **CNN-LSTM-BiLSTM**
- Test Accuracy: **96.20%**
- Lowest Type-I Error Rate: **2.23%**
