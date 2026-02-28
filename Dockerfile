FROM python:3.11-slim

# Configure a non-root user specifically for HF Spaces
RUN useradd -m -u 1000 user

USER user

# Set home to the user's home directory
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

# Change working directory
WORKDIR $HOME/app

# Install dependencies using the user
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the app files into the home directory, setting proper ownership
COPY --chown=user . $HOME/app

# Expose the standard port
EXPOSE 7860

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
