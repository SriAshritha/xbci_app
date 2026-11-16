# NOT tested in the build environment of the authors - treat as a starting point.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV XBCI_ARTIFACT_ROOT=/data/XBCI_BNCI2014_001
EXPOSE 8501
# mount your artifact folder:  docker run -p 8501:8501 -v C:\path\to\XBCI_BNCI2014_001:/data/XBCI_BNCI2014_001 xbci
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
