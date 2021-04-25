sudo docker build --tag flask-docker-base-app .
sudo docker run --rm --name flask-docker-base-app -p 5000:5000 flask-docker-base-app