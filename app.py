import os
import logging
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin
from src.utils.main_utils import decodeImage, encodeImageIntoBase64
import shutil

APP_HOST = "0.0.0.0"
APP_PORT = 8080

app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()

def run_detection(weights, source):
    os.system(f"python yolov5/detect.py --weights {weights} --img 640 --conf 0.5 --source {source}")
    return "yolov5/runs/detect/exp/inputImage.jpg"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
@cross_origin()
def predict_route():
    try:
        image = request.json.get('image')
        if not image:
            raise ValueError("Image data not found in the request")

        file_path = decodeImage(image, clApp.filename)
        print(file_path,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<>>>>>>>>>')
        best = os.path.abspath("yolov5/runs/train/yolov5s_results/weights/best.pt")
        result_image_path = run_detection(f"{best}", f"{file_path}")
        print(result_image_path,'<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<>>>>>>>>>')

        opencodedbase64 = encodeImageIntoBase64(result_image_path)
        result = {"image": opencodedbase64.decode('utf-8')}
        #os.system("rm -rf yolov5/runs")
        shutil.rmtree("yolov5/runs/detect")
        return jsonify(result)

    except ValueError as val:
        logger.error(val)
        return Response("Value not found inside JSON data", status=400)
    except KeyError as ke:
        logger.error(ke)
        return Response("Key value error, incorrect key passed", status=400)
    except Exception as e:
        logger.exception("An error occurred")
        return Response("Internal Server Error", status=500)

@app.route("/live", methods=['GET'])
@cross_origin()
def predict_live():
    try:
        result_image_path = run_detection("yolov5/runs/train/yolov5s_results/weights/best.pt", "0")
        #os.system("rm -rf yolov5/runs")
        shutil.rmtree("yolov5/runs/detect")
        return "Camera starting!!"

    except ValueError as val:
        logger.error(val)
        return Response("Value not found inside JSON data", status=400)
    except Exception as e:
        logger.exception("An error occurred")
        return Response("Internal Server Error", status=500)

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=True)
