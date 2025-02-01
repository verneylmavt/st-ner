# 📑 Named Entity Recognition (NER) Model Collections

This repository contains machine learning models of Named Entity Recognition (NER), designed to be deployed using ONNX and utilized in a Streamlit-based web application. The app provides an interactive interface for performing this task using neural network architectures. [Check here to see other ML tasks](https://github.com/verneylmavt/ml-model).

For more information about the training process, please check the `ner.ipynb` file in the `training` folder.

## 🎈 Demo App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://verneylogyt-ner.streamlit.app/)

![Demo GIF](https://github.com/verneylmavt/st-ner/blob/main/assets/demo.gif)

If you encounter message `This app has gone to sleep due to inactivity`, click `Yes, get this app back up!` button to wake the app back up.

<!-- [https://verneylogyt.streamlit.app/](https://verneylogyt.streamlit.app/) -->

## ⚙️ Running Locally

If the demo page is not working, you can fork or clone this repository and run the application locally by following these steps:

<!-- ### Prerequisites

Ensure you have the following installed:

- Python 3.8 or later
- pip (Python Package Installer)

### Installation Steps -->

1. Clone the repository:

   ```bash
   git clone https://github.com/verneylmavt/st-ner.git
   cd st-ner
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

Alternatively you can run `jupyter notebook demo.ipynb` for a minimal interface to quickly test the model (implemented w/ `ipywidgets`).

## ⚖️ Acknowledgement

I acknowledge the use of the **CoNLL-2003** dataset provided by the **Conference on Computational Natural Language Learning (CoNLL)**. This dataset has been instrumental in conducting the research and developing this project.

- **Dataset Name**: CoNLL-2003
- **Source**: [https://www.aclweb.org/anthology/W03-0419/](https://www.aclweb.org/anthology/W03-0419/)
- **Description**: This dataset was introduced as part of the CoNLL-2003 shared task on language-independent named entity recognition. It includes annotated data for four types of named entities: persons (PER), locations (LOC), organizations (ORG), and miscellaneous names (MISC). The dataset covers English and German languages and is widely used for training and evaluating NER systems.

I deeply appreciate the efforts of the CoNLL organization in making this dataset available.
