FROM python:3.12-slim

# Install ffmpeg for MP3 support
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY pyproject.toml README.md LICENSE ./
COPY harmonydagger/ harmonydagger/
RUN pip install --no-cache-dir -e ".[streamlit]"

# Copy demo app
COPY streamlit_app.py .

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
