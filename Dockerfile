FROM python:3.12

ENV PYTHONUNBUFFERED 1

WORKDIR /code

COPY requirements.txt /code/

RUN pip install --no-cache-dir -r requirements.txt

COPY app /code/

RUN python manage.py collectstatic
RUN python translator/migrations/0001_initial.py