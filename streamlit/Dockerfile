# Use the official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

RUN pip install requests
RUN pip install streamlit
RUN pip install streamlit-advanced-audio
RUN pip install streamlit-3d
RUN pip install httpx
RUN pip install dotenv
RUN pip install scipy
RUN pip install pydub
RUN pip install soundfile
RUN pip install google-auth
RUN pip install authlib


# Add ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# Copy files
COPY . .

# Expose port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
