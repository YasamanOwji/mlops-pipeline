# 🚀 Production-Ready MLOps Pipeline Template

> **Use this as a starter template for your own ML projects.**  
> It covers Data Versioning (DVC), Experiment Tracking & Registry (MLflow), Model Serving (FastAPI), Containerization (Docker), CI/CD (GitHub Actions), Orchestration (Prefect), and Monitoring (Evidently) — all connected out-of-the-box.
>
> **👥 For Teams**: Skip the boilerplate. Clone this repo, plug in your own dataset and model logic, and go from notebook to production in hours, not weeks.

[![CI/CD](https://github.com/YasamanOwji/mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/YasamanOwji/mlops-pipeline/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🏗️ Architecture Diagram

```mermaid
flowchart TD
    A[📁 Raw Data CSV] -->|dvc add & push| B[🗄️ DVC Remote Storage]
    A -->|Load & Split| C[🧪 Train/Test Split]
    
    subgraph MLflow_Tracking [MLflow Tracking Server]
        D[🏃 Run Training]
        D -->|Log Params/Metrics| E[📊 MLflow UI]
        D -->|Register Model| F[📦 Model Registry]
    end

    C --> D
    D -->|Artifact| G[🤖 RandomForest Model]
    
    F -->|Promote to Staging/Prod| H[📂 Deployment Folder]
    H -->|Load Model| I[⚡ FastAPI Server]
    
    I --> J[🐳 Docker Container]
    J --> K[☁️ Deploy to Cloud/VM]
    
    H -->|Reference Data| L[🔍 Evidently Monitoring]
    L -->|Drift/Quality Report| E

    subgraph CI_CD [GitHub Actions CI/CD]
        M[⬆️ Push to main] --> N[🧪 Run Pytest]
        N -->|Pass| O[🐳 Build Docker Image]
        O -->|Push| P[📦 Docker Registry]
    end

    I -.->|Trigger on Push| M