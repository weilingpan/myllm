FROM python:3.12

WORKDIR /app


COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src .

# docker build -t mymem0 .
# docker run -it --name mymem0 mymem0 /bin/bash
# docker exec -it mymem0 /bin/bash