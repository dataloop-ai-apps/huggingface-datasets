FROM hub.dataloop.ai/dtlpy-runner-images/cpu:python3.12_opencv

# Install the required packages
RUN pip install datasets git+https://github.com/dataloop-ai-apps/dtlpy-converters 'Pillow>=9.4.0'
#docker build --no-cache -t gcr.io/viewo-g/piper/agent/cpu/huggingface_datasets:0.0.3 -f Dockerfile .