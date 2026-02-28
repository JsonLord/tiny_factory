FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Expose the standard port
EXPOSE 7860

# Configure a non-root user specifically for HF Spaces
RUN useradd -m -u 1000 user

USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Change working directory
WORKDIR $HOME/app

# Copy the app files into the home directory, setting proper ownership
COPY --chown=user . $HOME/app

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
