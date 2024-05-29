FROM cpu.py3.8.opencv4.7

# Install the required packages
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt