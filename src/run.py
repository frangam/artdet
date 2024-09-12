from flask.helpers import send_from_directory
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
from os import remove
import os
import cv2
# from flask_ngrok import run_with_ngrok
from pyngrok import ngrok

from datetime import timedelta
import utils
from keras import backend as K

#!pip install pyngrok==4.1.1
# !ngrok authtoken "2GZuKvg2gVgoOnoWyyTog01UDEY_54RgkWDSxGMhcGEVf4Vpp"


K.clear_session()

# Configuration of allowed file extensions.
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'JPG', 'PNG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)

# Set the expiration time for static files cache.
app.send_file_max_age_default = timedelta(seconds=1)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detection", methods=["POST"])
def detection():
    print("detection")
    
    # Verificar si el archivo existe en la solicitud
    if 'file' not in request.files:
        return jsonify({"error": 1002, "msg": "No file part in the request"}), 400

    f = request.files['file']
    
    # Verificar si se seleccionó un archivo
    if f.filename == '':
        return jsonify({"error": 1003, "msg": "No selected file"}), 400

    # Validar el tipo de archivo permitido
    if not allowed_file(f.filename):
        return jsonify({"error": 1001, "msg": "Invalid file type. Only PNG, JPG, and JPEG are allowed"}), 400

    # Usar secure_filename para evitar problemas de ruta
    filename = secure_filename(f.filename)
    images_folder = 'static/images/'
    upload_path = os.path.join(images_folder, filename)

    # Verificar si el directorio de imágenes existe, si no, crearlo
    print("images_folder:", images_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    try:
        # Guardar el archivo en el servidor
        f.save(upload_path)

        # Leer la imagen usando OpenCV
        img = cv2.imread(upload_path)
        if img is None:
            return jsonify({"error": 1004, "msg": "Unable to read the uploaded image"}), 400

        # Guardar la imagen procesada (aquí puedes realizar alguna modificación si es necesario)
        current_upload = os.path.join(images_folder, 'test.jpg')
        cv2.imwrite(current_upload, img)

        return render_template("damage_detection.html", name_file=filename)
    
    except Exception as e:
        return jsonify({"error": 1005, "msg": str(e)}), 500
    


@app.route('/home/artworks/damage')
def show_damage():
    K.clear_session()
    name = request.args.get('name')


    images_folder = 'static/images/'
    box_path = images_folder + name

    # Once we have the image, we detect the damages
    zones = utils.saveImage_GetFeatures(name)

    image_size = zones[0]
    undamaged_zone = zones[1]
    damaged_zone = zones[2]
    percentage = zones[3]

    remove(box_path)

    return render_template("show_damage.html", name_file=name, size=image_size, undamaged_zone=undamaged_zone,
                           damaged_zone=damaged_zone, percentage=percentage)

@app.route('/home/artworks/damage/download')
def download_images():
    pathfile = 'static/images/'
    path = 'detection.jpg'
    return send_from_directory(pathfile, path, as_attachment=True)


if __name__ == "__main__":
    # Inicia el túnel de ngrok y obtén la URL pública
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel: {public_url}")

    # Ejecuta la app Flask
    app.run(port=5000)
