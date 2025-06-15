FROM python:3.13-slim

WORKDIR /app

COPY . .

# 🛠 Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "agent"]
