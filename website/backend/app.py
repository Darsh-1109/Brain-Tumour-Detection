from flask import Flask, render_template, request, redirect, send_file, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from flask_cors import CORS
import os
import shutil
from ultralytics import YOLO
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import tensorflow as tf

model = YOLO(r'C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\models\yolov8n-seg-d3.pt')

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tumour_detections.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = r'C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\website\backend\mri_scans'

db=SQLAlchemy()

class File(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(500), nullable=False)
    sender_name = db.Column(db.String(500), nullable=False)
    description = db.Column(db.String(500), nullable=False)
    tumour_type = db.Column(db.String(500), nullable=True)
    affected_body_functionality = db.Column(db.String(500), nullable=True)
    segmented_image = db.Column(db.String(500), nullable=True)
    date_created = db.Column(db.DateTime, default=datetime.utcnow)
        
    def __repr__(self) -> str:
        return f"{self.id} - {self.filename}"

db.init_app(app)
app.app_context().push()
db.create_all()

@app.route('/tumour_upload', methods=['GET', 'POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    description = request.form.get('description')
    sender_name = request.form.get('sender_name')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    

    allowed_extensions = {'jpg', 'jpeg', 'png','pdf', 'doc', 'docx'}
    file_extension = file.filename.split('.')[-1].lower()
    
    if file_extension in allowed_extensions:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        results=model(filename, save=True, project=r'C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\website\backend\tumour_detections', name='Detection of '+file.filename)
        
        # print(results)
        
        # for result in results:
        #     if result.masks==None:
        #         class_name='No Tumour'
        #     else:
                # class_name=result.names[0]
                
        # Load the model using tf.keras
        classification_model = tf.keras.models.load_model(r'C:\Users\a21ma\OneDrive\Desktop\Code\Projects\Brain Tumour Detection (IPD)\models\tumour_classification_model.h5')
        
        img = image.load_img(filename, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # img_array = preprocess_input(img_array)
        
        # Use the loaded model to predict the class of the new image
        predictions = classification_model.predict(img_array)
        class_name = np.argmax(predictions)

        if class_name==0:
            class_name='Glioma Tumour'
            body_functionality='Brain, Nerves, Vessels'
        elif class_name==1:
            class_name='Meningioma Tumour'
            body_functionality='Brain, Spinal Cord, Sleep'
        elif class_name==2:
            class_name='No Tumour'
            body_functionality='Normal'
        elif class_name==3:
            class_name='Pituitary Tumour'
            body_functionality='Thyroid, Vision, High BP, High Blood Sugar'
                    
                
        
        new_file = File(filename=file.filename, sender_name=sender_name, description=description,tumour_type=class_name, affected_body_functionality=body_functionality, segmented_image=file.filename) 
        
        db.session.add(new_file)
        db.session.commit()
        
        return jsonify({'message': 'File uploaded successfully'})
    else:
        return jsonify({'error': 'Invalid file type'}), 400
    
@app.route('/tumour_get_uploaded_files', methods=['GET'])
def get_uploaded_files():
    files = File.query.all()
    files_data = [{'id': file.id, 'filename': file.filename, 'sender_name': file.sender_name, 'description': file.description, 'tumour_type': file.tumour_type, 'affected_body_functionality': file.affected_body_functionality, 'segmented_image':file.segmented_image,'date_created': file.date_created} for file in files]
    return jsonify(files_data)

@app.route('/tumour_uploads/<filename>', methods=['GET'])
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/segmented_uploads/<segmented_image>', methods=['GET'])
def serve_segmented_file(segmented_image):
    return send_from_directory('C:/Users/a21ma/OneDrive/Desktop/Code/Projects/Brain Tumour Detection (IPD)/website/backend/tumour_detections/Detection of '+segmented_image, segmented_image)

@app.route('/tumour_delete_file/<int:id>', methods=['DELETE'])
def delete_file(id):
    file_to_delete = db.session.get(File, id)
    if file_to_delete:
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_to_delete.filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            shutil.rmtree('C:/Users/a21ma/OneDrive/Desktop/Code/Projects/Brain Tumour Detection (IPD)/website/backend/tumour_detections/Detection of '+file_to_delete.filename)
        
        db.session.delete(file_to_delete)
        db.session.commit()
        return jsonify({'message': 'File deleted successfully'})
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)
