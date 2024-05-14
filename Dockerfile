FROM python:3.9

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN pip install fastapi
RUN pip install uvicorn

# 
COPY . .

RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN pip install python-multipart

# 
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=80"]