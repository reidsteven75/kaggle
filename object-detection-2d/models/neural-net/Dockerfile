FROM tensorflow/tensorflow:latest-py3

# Set working dir
WORKDIR /code

# Install packages
COPY requirements*.txt /code/
RUN pip3 install -r requirements.txt 

# Copy code
COPY . /code

CMD [ "python3", "./index.py" ]