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
📄 **Detailed Architecture, Problem Statement & Technical Design**  
👉 *(Link to Google Doc – (https://docs.google.com/document/d/1F8qbUlpY5mmD9qXqkOsk9WE-dmcd-b-PoYVR8xLBuTM/edit?usp=sharing))*
We present an end-to-end maritime intelligence system that tackles illegal, unreported, and unregulated (IUU) fishing using satellite radar, computer vision, and behavioral analysis.
Our approach goes beyond simple vessel detection by fusing Sentinel-1 SAR imagery, physics-based ship detection, CNN-based visual validation, AIS silence verification, and context-aware risk scoring to identify dark vessels operating inside protected marine regions. By combining spatial legality (MPAs), motion cues, fleet-level context, and explainable risk fusion, the system produces actionable, analyst-ready intelligence rather than raw alerts — enabling faster, more informed maritime enforcement decisions.

### ☁️ AWS Track

📄 **AWS/Kiro Architecture, Execution Flow & Design Rationale**  
👉 *(Link to Google Doc – https://docs.google.com/document/d/1zZNevPwG6epvq0aSeYlOOPLbUZ6hpRcabT4RDZEJY1E/edit?usp=sharing)*
Our system leverages Kiro as the central orchestration and execution layer for the pipeline, enabling seamless coordination of satellite ingestion, model execution, and risk analysis workflows. Kiro is used to manage task scheduling, modular pipeline stages, and controlled execution of compute-intensive components, allowing the system to remain scalable, reproducible, and easy to extend. By structuring the entire intelligence flow as Kiro-managed tasks, we demonstrate how Kiro can be effectively used to operationalize complex AI pipelines — from data ingestion to final threat assessment — with clarity and reliability.

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

## 🤝 Contributors

1. [Varun E](https://github.com/varun-iiitb)
2. [Potini Sahiti](https://github.com/sahiti3636)
3. [Navya Sharma](https://github.com/navya2208)
4. [Shivansh Shah](https://github.com/shivansh-shah)

---

**Developed for HackXios 2025**
