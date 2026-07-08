FROM python:3.9-slim

WORKDIR /app

# کپی فایل‌های مورد نیاز
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی سورس کد و کانفیگ
COPY src/ src/
COPY config/ config/

# در حالت واقعی، مدل از قبل در staging/production وجود دارد
# ولی برای دمو، یک مسیر پیش‌فرض می‌سازیم
RUN mkdir -p deployment/production/model

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]