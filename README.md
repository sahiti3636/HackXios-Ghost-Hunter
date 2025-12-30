# 👻 Ghost Hunter: Dark Vessel Detection System

![Ghost Hunter](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![React](https://img.shields.io/badge/Frontend-React%20%2F%20Next.js-blueviolet)

> **Advanced maritime intelligence system fusing Sentinel-1 SAR imagery with GenAI to detect illicit "dark vessels".**

---

## 🎥 Demo

[![Watch the Demo](https://img.youtube.com/vi/PLACEHOLDER_VIDEO_ID/maxresdefault.jpg)](https://youtu.be/PLACEHOLDER_VIDEO_ID)

*(Click above to watch the project walkthrough)*

---

## 📚 Tracks & Documentation

### 🏆 Open Innovation Track
We leverage cutting-edge **Generative AI (Gemini 1.5)** and **Computer Vision** to solve a critical global security challenge: illegal, unreported, and unregulated (IUU) fishing. Our system moves beyond simple detection to provide *actionable intelligence* and automated threat assessment.

### ☁️ AWS Track
Our architecture is designed to scale on AWS infrastructure, utilizing **S3** for satellite data storage, **EC2/SageMaker** for heavy model inference, and **Lambda** for lightweight API orchestration.

---

## ⚠️ Deployment & Architecture Note

**Why is this not hosted on Vercel/Render?**

Ghost Hunter deals with **high-resolution Synthetic Aperture Radar (SAR) satellite imagery** and runs **complex CNN inference** locally. 
1.  **Computational Intensity**: Processing 250MB+ satellite product files requires significant RAM and CPU power, exceeding the limits of standard free-tier PaaS functionality.
2.  **Model Size**: The PyTorch CNN models and geospatial libraries (GDAL/Rasterio) create a large build slug that is best managed in a containerized environment (Docker/Kubernetes) or a dedicated high-performance instance.

**We provide a seamless local setup script to replicate the full production environment on your machine.**

---

## 🚀 Key Features

-   **Multi-Source Data Fusion**: Correlates Sentinel-1 SAR imagery with real-time AIS data.
-   **Dark Vessel Detection**: Identifies vessels visible in radar but missing from AIS tracking.
-   **Risk Assessment**: Assigns risk scores based on behavior, location (e.g., inside MPAs), and history.
-   **GenAI Intelligence**: Generates automated intelligence reports using Gemini 1.5 Pro/Flash.
-   **Interactive Dashboard**: A modern Next.js frontend for visualizing detection results on a map.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.12+**: [Download Python](https://www.python.org/downloads/)
2.  **Node.js & npm**: [Download Node.js](https://nodejs.org/) (Required for the frontend)
3.  **Git**: [Download Git](https://git-scm.com/)

---

## ⚡ Quick Start (Automated)

We provide automated scripts to set up the environment, install dependencies, and launch the application in one go.

### 🍎 macOS / Linux

1.  Open your terminal.
2.  Clone the repository:
    ```bash
    git clone https://github.com/sahiti3636/HackXios-Ghost-Hunter.git
    cd HackXios-Ghost-Hunter
    ```
3.  Run the setup script:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```

### 🪟 Windows

1.  Open PowerShell.
2.  Clone the repository:
    ```powershell
    git clone https://github.com/sahiti3636/HackXios-Ghost-Hunter.git
    cd HackXios-Ghost-Hunter
    ```
3.  Run the setup script:
    ```powershell
    .\setup.ps1
    ```

---

##  Manual Setup

### 1. Backend
```bash
python -m venv venv
# Activate venv (source venv/bin/activate OR .\venv\Scripts\Activate)
pip install -r requirements_backend.txt
python app.py
```

### 2. Frontend
```bash
cd ghost-hunter-frontend
npm install
npm run dev
```

---

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Commit your changes.
4.  Push and open a Pull Request.

---

**Developed for HackXios 2025**
