FROM python:3.7-slim
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
#Explicitly installing pytorch for CPU
RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
EXPOSE 5000
ENTRYPOINT [ "python3","main.py" ]