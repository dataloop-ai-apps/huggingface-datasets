FROM dataloopai/dtlpy-agent:cpu.py3.8.opencv4.7

# Install the required packages
RUN pip install datasets git+https://github.com/dataloop-ai-apps/dtlpy-converters 'Pillow>=9.4.0'