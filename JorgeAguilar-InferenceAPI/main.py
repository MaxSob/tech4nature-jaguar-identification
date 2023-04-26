from flask import Flask, request,jsonify
import onnxruntime
import json
from prepare_img import read_load_image, preprocess_img

model_db = [{"id":0,
            "model":"Jaguar Prediction CNN",
            "path_v1":"models/onnx/wHC_1.onnx",
            "path_v2":"models/onnx/wHC_1_1.onnx" },
            {"id":1,
            "model":"Object Detection CNN",
            "path_v1":"models/onnx/model_bbox_regression_and_classification_m1_vf.onnx",
            "path_v2":"models/onnx/model_bbox_regression_and_classification_m2_vf.onnx"},
            {"id": 2,
            "model":"RESNET50",
            "path":"Jaguar identification"}]

session0 = onnxruntime.InferenceSession(model_db[1]["path_v1"])
session1 = onnxruntime.InferenceSession(model_db[0]["path_v1"])

app =Flask(__name__)

@app.route('/')
def welcome():
    return"welcome models"



#Allow files with extension png, jpg and jpeg
ALLOWED_EXTENSIONS = set(['jpg' , 'jpeg' , 'png', 'JPG','JPEG','PNG'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


#create the first route for the model object detection 
@app.route('/models/object_detection', methods = ["GET", "POST"])
def object_detection():
    if request.method== "POST":
        #load the json data
        img_data= json.loads(request.data)
        #Error checking if there exist a file and if the filename is an empty string
        if img_data is None or img_data["filename"]=="":
            #if not exist a file the it will return an error
            return jsonify({"error": "No file"})
        else:
            #Now, if there exist a file, then verify if the file is in allowed file
            if allowed_file(img_data["filename"]):
                #read the image
                test_img =  read_load_image(img_data["file_path"]+img_data["filename"])
                
                # Define input and output names
                input_name = session0.get_inputs()[0].name
                output_name = session0.get_outputs()[0].name
                #insert the img
                input_data = {input_name: test_img}
                #run session
                outputs = session0.run([output_name], input_data)
                #create a list 
                output = [float(outputs[0][0][0]), float(outputs[0][0][1]), float((outputs[0][0][2])), float(outputs[0][0][3])]
                
                return{"Resultados":output}
            else:
                #return error
                return jsonify({"error": "The chossen file is not allowed"})
    
    return "Inference API"

#create the first route for the model object detection 
@app.route('/models/jaguar_prediction', methods = ["GET", "POST"])
def object_detection():
    session1 = onnxruntime.InferenceSession('models/onnx/model_bbox_regression_and_classification_m1_vf.onnx')

    if request.method== "POST":
        #load the json data
        img_data= json.loads(request.data)
        #Error checking if there exist a file and if the filename is an empty string
        if img_data is None or img_data["filename"]=="":
            #if not exist a file the it will return an error
            return jsonify({"error": "No file"})
        else:
            #Now, if there exist a file, then verify if the file is in allowed file
            if allowed_file(img_data["filename"]):
                #read the image
                test_img =  preprocess_img(img_data["file_path"]+img_data["filename"])
                # Define input and output names
                input_name = session1.get_inputs()[0].name
                output_name = session1.get_outputs()[0].name
                #insert the img
                input_data = {input_name: test_img}
                #run session
                outputs = session1.run([output_name], input_data)
                #create a list 
                #output = [float(outputs[0][0][0]), float(outputs[0][0][1]), float((outputs[0][0][2])), float(outputs[0][0][3])]
                
                return{"Resultados":str(outputs)}
            else:
                #return error
                return jsonify({"error": "The chossen file is not allowed"})
    
    return "Inference API"