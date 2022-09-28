from dataclasses import field
from flask_restx import Resource, reqparse, fields, marshal
from flask_app import swagger_api
from flask import jsonify


from utils.TensorflowInference import TensorflowInference

ModelInferenceArguments = reqparse.RequestParser()
# ModelInferenceArguments.add_argument('Authorization',required=False,help='provide Authorization',location='headers')
ModelInferenceArguments.add_argument('base64String',help='provide base64 encoded image', required=True)

ns = swagger_api.namespace('ModelInference', description='ModelInference')
swagger_api.add_namespace(ns)


from PIL import Image
import numpy as np
from io import BytesIO
import base64
import cv2

TensorflowInference_handler = TensorflowInference(modelPath="./modeldata/buddha_keras_model.h5", LabelsPath="./modeldata/labels.txt")
# Load the model

Model_InferenceResponseParams = swagger_api.model('Model_InferenceSuccess', {
    'inferenceLabel': fields.String,
    'inferenceConfidence': fields.String,
    'error': fields.Boolean,
    'error_msg': fields.String,
    'base64image': fields.String
})

@ns.expect(ModelInferenceArguments)
class ModelInferenceAPI(Resource):

    AuthorizationToken = "12345"

    def LocalInference(self, base64String, inferenceData={}):


        if "," in base64String:
            base64String = base64String.split(",")[1]

        f = base64.b64decode(base64String)
        f = np.frombuffer(f, dtype=np.uint8)
        image = cv2.imdecode(f, cv2.IMREAD_UNCHANGED)

        h,w,c = image.shape
        if c>3:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        Pred_label, PredConfidence = TensorflowInference_handler.infer(image, isopencvImage=True)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image)
        buff = BytesIO()
        pil_img.save(buff, format="JPEG")
        webimg = base64.b64encode(buff.getvalue()).decode("utf-8")

        inferenceData["base64image"]= webimg
        inferenceData["error"] = False
        inferenceData["error_msg"]= ""
        inferenceData["inferenceLabel"]= Pred_label
        inferenceData["inferenceConfidence"]= PredConfidence
        
        return inferenceData

    def __validate_auth_key(self, authkey):

        if "Bearer" in authkey:
            Token = authkey.split(" ")[1]
            if Token==ModelInferenceAPI.AuthorizationToken:
                return True
        return False


    @ns.response(model=Model_InferenceResponseParams, code=200, description="Success")
    def post(self):

        return_data = {
            "error": False,
            "error_msg": ""
        }
        
        ParsedArgs = ModelInferenceArguments.parse_args()
        File = ParsedArgs['base64String']
        return_data = self.LocalInference(base64String=File, inferenceData=return_data)

        return marshal(return_data, Model_InferenceResponseParams,envelope='resource'),200