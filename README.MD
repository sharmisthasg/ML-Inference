Steps to check Flask server:

1. Run `pip install requirements.txt`
2. Run `python main.py`
3. Check output at `0.0.0.0:5000/` port

Steps to create and run docker containers:

1. Build docker image using `sudo docker build --tag flask-docker-base-app .`
2. Run container using `sudo docker run --name flask-docker-base-app -p 5000:5000 flask-docker-base-app`
3. Check output at `0.0.0.0:5000/` port


Ensure that docker is installed and running on your system 