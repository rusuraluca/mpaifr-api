import torch

from flask import Flask, request, jsonify, Response
from config import CONFIG
from aifr.models.multitask_dal.model import Multitask_DAL
from utils.similarity_handler import SimilarityHandler


app = Flask(__name__)

config = CONFIG.instance()
model = Multitask_DAL(embedding_size=512, number_of_classes=500, margin_loss_name=config['margin_loss'])
if config["model_weights"]:
    model.load_state_dict(torch.load(config["model_weights"], map_location='cpu'))
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

    def load_image_from_file(file, transformation):
                image = Image.open(file).convert('RGB')
                if transformation is not None:
                    image = transformation(image)
                    image = image.unsqueeze(0)
                return image

    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

    image1 = load_image_from_file(file1, transform)
    image2 = load_image_from_file(file2, transform)

    #with torch.no_grad():
    #    features1 = model(image1, return_embeddings=True)
    #    features2 = model(image2, return_embeddings=True)
    #similarity = functional.cosine_similarity(features1, features2)
    similarity = 123

    if similarity:
        return jsonify({"similarity": round(similarity, 2)})

    return jsonify({"request_error": "The parameters were valid but the request failed."}), 402


@app.route('/batch_images_similarity', methods=['POST'])
def get_batch_similarity():
    if 'imageList1' not in request.files or 'imageList2' not in request.files:
        return jsonify({"invalid_request_error": "Please provide the images."}), 400

    image_list_1_files = request.files.getlist('imageList1')
    image_list_2_files = request.files.getlist('imageList2')

    if not image_list_1_files:
        return jsonify({"invalid_request_error": "Please provide a list of images."}), 400
    if not image_list_2_files:
        return jsonify({"invalid_request_error": "Please provide a second list of images."}), 400

    similarities = []

    for file1 in image_list_1_files:
        for file2 in image_list_2_files:
            if file1 and file2:
                similarity = SimilarityHandler.get_response(model, file1, file2)
                similarities.append(similarity)
            else:
                return jsonify({"invalid_request_error": "Invalid files provided."}), 400

    if similarities:
        average_similarity = sum(similarities) / len(similarities)
        return jsonify({"similarity": round(average_similarity, 2)})

    return jsonify({"request_error": "The parameters were valid but the request failed."}), 402

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, threaded=True)
