# Beyond accuracy: establishing a realistic performance baseline and uncovering core challenges of large language models in grading unstructured liver pathology reports

This repository contains the official code and analysis scripts for the research paper, "Beyond accuracy: establishing a realistic performance baseline and uncovering core challenges of large language models in grading unstructured liver pathology reports".

## Abstract

> Large language models (LLMs) have demonstrated exceptional performance in processing structured or semi-structured medical texts. However, their true capabilities, limitations, and failure modes in interpreting highly unstructured and semantically ambiguous narrative pathology reports remain unclear. This knowledge gap hinders the reliable application of LLMs in diagnostic domains requiring advanced clinical reasoning, such as pathology. This study aims to: (1) develop and validate a framework for the automated grading of unstructured liver pathology reports using open-source LLMs; (2) systematically evaluate the performance of over 20 mainstream LLMs on three key pathological indicators; (3) uncover the macroscopic challenges of current LLMs in simulating expert-level clinical reasoning through an in-depth error analysis; and (4) investigate how the intrinsic characteristics of models (e.g., scale, architecture, and reasoning strategies) impact their performance.

## Repository Structure

This repository is organized into a series of scripts that handle data preparation, model evaluation, and results analysis.

*   `00_data_preparation.py`: Prepares the dataset by splitting it into training, validation, and test sets. This is the first script that needs to be run.
*   `07_evaluation_llamacpp_linux_v1.1.py`: The main, automated script for running the systematic evaluation of all models using the `llama.cpp` server backend. It programmatically starts the server for each model, runs the evaluation across all tasks, and saves the results.
*   `09_draw_task_*.py`: A collection of scripts used to generate the final tables and figures presented in the paper from the raw and aggregated results produced by the evaluation scripts.
*   `results/`: This directory will contain all generated reports, summary files, and plots.
*   `logs/`: This directory will contain detailed logs for each script run.

## Setup

1.  Clone the repository to your local machine:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  Install the required Python dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Data

Due to patient privacy and ethical considerations, the clinical dataset used in this study cannot be made publicly available. As stated in the paper, the dataset is available from the corresponding author upon reasonable request.

Before running the evaluation, you must first run the data preparation script on your local dataset (`data.json`):
```bash
python 00_data_preparation.py
```

## Usage

The main evaluation is orchestrated by the `07_evaluation_llamacpp_linux_v1.1.py` script. This script is designed to run the full suite of experiments automatically.

Before running, you must configure the paths inside the script:

1.  Open `07_evaluation_llamacpp_linux_v1.1.py` in a text editor.
2.  Modify the `LLAMACPP_SERVER_EXE` variable to point to the location of your `llama-server` executable.
3.  Modify the `MODELS_DIR` variable to point to the directory containing your `.gguf` model files.
4.  Customize the `MODELS_TO_EVALUATE` list to include the models you wish to test.

Once configured, run the script from your terminal:
```bash
python 07_evaluation_llamacpp_linux_v1.1.py
```
The script will automatically start and stop the `llama.cpp` server for each model and save all results to the `results/` directory.