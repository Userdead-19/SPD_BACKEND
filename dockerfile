FROM python:3.10

WORKDIR /App

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["fastapi","dev","index.py"]