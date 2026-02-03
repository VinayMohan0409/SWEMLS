FROM python:3.12-slim

# Minimal runtime deps (torch will pull numpy etc)
WORKDIR /app

# System deps for torch wheels / general robustness
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code
COPY aki_service /app/aki_service

# Optional: copy model artifacts (user should place aki_model.pt here)
COPY model /app/model

ENV PYTHONUNBUFFERED=1

# Grader will provide MLLP_ADDRESS and PAGER_ADDRESS
CMD ["python", "-m", "aki_service"]
