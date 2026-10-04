# Multi-Domain Image Restoration & Face-to-Sketch Synthesis

This repository contains the complete implementation for **Generative AI Assignment #1**. It encompasses four unified generative AI systems: a universal denoising autoencoder, a hard-routed specialist system, a jointly trained Soft Mixture-of-Experts (MoE), and a style-conditioned Conditional GAN (Pix2Pix-based).

## 📂 Repository Structure

* `App/` - Contains the full production-ready Web Application.
  * `app.py` - Stateless FastAPI backend for ONNX inference.
  * `index.html` - Google Stitch-prototyped React/Tailwind/Vanilla frontend.
  * `Dockerfile` & `docker-compose.yml` - Containerization configurations.
  * `requirements.txt` - Dependency file for the deployment environment.
* `notebooks/` - Contains all Kaggle scripts for data-preparation, Optuna hyperparameter studies, training loops, evaluation scripts, and ONNX export code.
* `models/` - Directory for storing the trained ONNX models (ignored by Git to prevent large file uploads).
* `IEEE_Technical_Report.tex` - The academic IEEE-formatted technical report source code.

## 🚀 Model Weights Download

In accordance with GitHub size limits, the large ONNX model files are not hosted directly in this repository. 
**Please download the pre-trained models from the following link:**
👉 `[INSERT YOUR GOOGLE DRIVE / KAGGLE DOWNLOAD LINK HERE]`

Once downloaded, place the following files directly inside the `App/models/` directory:
* `universal_autoencoder.onnx`
* `corruption_classifier.onnx`
* `specialist_1.onnx`, `specialist_2.onnx`, `specialist_3.onnx`
* `soft_moe_system.onnx`
* `cgan_generator.onnx`

## ⚙️ Execution Instructions (Docker)

The complete application is containerized and can be launched with a single command. 

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
* The pre-trained ONNX models placed in `App/models/`.

### Startup Command
1. Clone the repository and navigate to the application folder:
   ```bash
   git clone https://github.com/MishalFatima09/genAI_A1.git
   cd genAI_A1/App
   ```
2. Build and start the containerized application:
   ```bash
   docker-compose up --build
   ```

### Accessing the Application
Once the container starts, open your web browser and navigate to:
**[http://localhost:8000](http://localhost:8000)**

You can upload corrupted images, capture live webcam frames, and evaluate all four tasks interactively. To gracefully shut down the server, press `Ctrl+C` in your terminal.
