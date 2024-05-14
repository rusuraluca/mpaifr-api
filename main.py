import torch
import os, sys
from flask import Flask, request, jsonify

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import CONFIG
from aifr.utils.model_handler import ModelHandler
from utils.similarity_handler import SimilarityHandler


app = Flask(__name__)

config = CONFIG.instance()
model = ModelHandler().get_model(config['model_name'], config['dataset'], config['margin_loss'])
if config["model_weights"]:
    model.load_state_dict(torch.load(config["model_weights"]), map_location=torch.device('cpu'))
    print(f'Loaded weights from {config["model_weights"]}')
model.eval()


@app.route('/')
def home():
    return "MPAIFR REST API"


@app.route('/images_similarity', methods=['POST'])
def get_similarity():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"invalid_request_error": "Please provide two images."}), 400

    file1 = request.files['image1']
    file2 = request.files['image2']

    if not file1 or not file2:
        return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    similarity = SimilarityHandler.get_response(model, file1, file2)

    if similarity:
        return jsonify({"similarity": round(similarity, 2)})

    return jsonify({"request_error": "The parameters were valid but the request failed."}), 402


@app.route('/batch_images_similarity', methods=['POST'])
def get_batch_similarity():
    if 'imageList1' not in request.files or 'imageList2' not in request.files:
        print("Please provide the images.")
        return jsonify({"invalid_request_error": "Please provide the images."}), 400

    image_list_1_files = request.files.getlist('imageList1')
    image_list_2_files = request.files.getlist('imageList2')

    if not image_list_1_files:
        print("Please provide a list of images.")
        return jsonify({"invalid_request_error": "Please provide a list of images."}), 400
    if not image_list_2_files:
        print("Please provide a second list of images.")
        return jsonify({"invalid_request_error": "Please provide a second list of images."}), 400

    similarities = []

    for file1 in image_list_1_files:
        for file2 in image_list_2_files:
            if file1 and file2:
                similarity = SimilarityHandler.get_response(model, file1, file2)
                similarities.append(similarity)
            else:
                print("Invalid files provided.")
                return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    if similarities:
        average_similarity = sum(similarities) / len(similarities)
        return jsonify({"similarity": round(average_similarity, 2)})

    print("The parameters were valid but the request failed.")
    return jsonify({"request_error": "The parameters were valid but the request failed."}), 402


if __name__ == '__main__':
    app.run(debug=True)
