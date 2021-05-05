Steps to create and run docker containers (ensure that the docker daemon is running):

1. Run the shell script `run_docker.sh` using the command `sh run_docker.sh`
2. Check if the server is running at `0.0.0.0:5000/` or `localhost:5000/`

Steps to check inference on images (once docker is running and previous step is executed):
1. Download the image to run inference on (eg. dog.jpg) in the `/images` directory
2. Run the command `curl http://localhost:5000/inference/dog.jpg` or any other image of your choice to get the prediction (class of object in the image)


Steps to check Flask server only without Docker Container (pip3 needs to be installed in your local machine for this):

1. Run `pip3 install -r requirements.txt`
2. Run `pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu -f https://download.pytorch.org/whl/torch_stable.html`
3. Run `python3 main.py`
4. Check if the server is running at `0.0.0.0:5000/` or `localhost:5000/`
5. Run the command `curl http://localhost:5000/inference/dog.jpg` or any other image of your choice to get the prediction (class of object in the image)
