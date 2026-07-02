FROM python:3.10-slim

# تسطيب الـ C-Compiler والمكتبات الأساسية المطلوبة لعملية بناء HDBSCAN و UMAP
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# السحر هنا: تنزيل نسخة الـ CPU فقط لتصغير حجم الـ Image وتخفيف التحميل
RUN pip install --no-cache-dir torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]