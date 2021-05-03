from flask import Flask
from ml_service import infer
app = Flask(__name__)

@app.route('/')
def hello():
	return "This is a test message"

@app.route('/inference/<image>')
def run_inference(image):
	try:
		print("image to be inferred is:",image)
		prediction = infer(image)
		return prediction
	except OSError as ose:
		return "File Not Found: Please load image, " + image + ", in images/ directory"
	except:
		return "Internal Server Error: Please contact ml-inference team 16"

if __name__ == "__main__":
	app.run(host ='0.0.0.0', port = 5000, debug = True) 