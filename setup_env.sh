#!/bin/bash
echo "🚀 Setting up MLOps Pipeline Template..."

# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialize DVC (if not already)
dvc init --no-scm  # استفاده از --no-scm برای اینکه با Git تداخل نکند

# 4. Create DVC remote (local folder for demo)
mkdir -p ../dvc-storage
dvc remote add -d local_remote ../dvc-storage

# 5. Create data folder and generate initial dataset
mkdir -p data
cd src && python generate_data.py && cd ..

echo "✅ Setup complete! Run 'cd src && python run_pipeline.py' to start."