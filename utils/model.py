import time
import os
import subprocess
import requests
import json
import cv2
import numpy as np
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow import make_tensor_proto, make_ndarray


class ModelServer:
    def __init__(self,model):
        self.MyOut = None
        self.stdout = None
        self.stderr = None

        self.model = model
        # Config details of model to be used for inference
        self.model_config = {
                            'face' : {'name':'face-detection-retail-0005', 'port':9001, 'in_key':'input.1', 'out_key':'527'},
                            'person' : {'name':'person-detection-retail-0013', 'port':9002, 'in_key':'data', 'out_key':'detection_out'},
                            'mask' : {'name':'face_mask', 'port':9003, 'in_key':'data', 'out_key':'fc5'},
                        }
        self.model_name = self.model_config[self.model]['name']
        self.port = self.model_config[self.model]['port']
        self.server_initialize()

    def server_initialize(self):
        self.CMD_Out = subprocess.Popen(
            ["docker", "ps", "-a", "--format", '"{{.Names}}"'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        self.stdout, self.stderr = self.CMD_Out.communicate()

        if str(self.stdout).find(self.model) < 0:
            print(f'[initializing {self.model} Model Server Container...]')
            cmd = f'docker run -d --name Model_Server_{self.model} -v "$(pwd)/model":/opt/ml -p {self.port}:9001 openvino/model_server /ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/{self.model_name} --model_name {self.model_name} --port 9001 --shape auto'
            os.system(cmd)
            time.sleep(5)
            print(f'[{self.model} Model Server initiallized]')
        else:
            print(f'[{self.model} Model Server is already initiallized]')

    def detect(self, img, thresh=0.5):

        '''
        Inference method for detection

        params
        img: raw image read with opencv/ raw frame from a video. (No preprocessing required)
        thresh: Threshold for minimum detection score.
        
        returns -> classes,scores,boxes
        classes: class of each detected object
        scores: detection probabilities 
        boxes: bounding boxes in (Xmin,Ymin,Xmax,Ymax) format, scaled to original input image size.(No postprocessing required)

        '''

        if self.model == 'person':
        	input_shape = (544,320)
        elif self.model == 'face':
        	input_shape = (300,300)
        else:
        	raise Exception('Loaded model is not an object detection model.')

        image = cv2.resize(img,input_shape)
        image = image.transpose(2,0,1)
        image = np.expand_dims(image,axis=0).astype(np.float32)

        out = self.infer(image)
        out = out[0,0]
        
        to_del = []
        for i,res in enumerate(out):
            if res[2]<=thresh:
                to_del.append(i)
        out = np.delete(out,to_del,axis=0)
        classes = out[:,1]
        scores = out[:,2]
        boxes = out[:,3:]

        boxes[:,0] = (boxes[:,0]*input_shape[1])*(img.shape[1]/input_shape[1])
        boxes[:,1] = (boxes[:,1]*input_shape[0])*(img.shape[0]/input_shape[0])
        boxes[:,2] = (boxes[:,2]*input_shape[1])*(img.shape[1]/input_shape[1])
        boxes[:,3] = (boxes[:,3]*input_shape[0])*(img.shape[0]/input_shape[0])

        return classes,scores,boxes.astype(np.int32)

    def check_mask(self, img):
        if self.model != 'mask':
            raise Exception('Loaded model is not mask model')

        image = cv2.resize(img,(224,224))
        image = image.transpose(2,0,1)
        image = np.expand_dims(image,axis=0).astype(np.float32)

        out = -1*self.infer(image)[0]
        out = False if out>0.65 else True

        return out

    def infer(self, my_image):

        channel = grpc.insecure_channel(f'localhost:{self.port}')
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        request = predict_pb2.PredictRequest()
        request.model_spec.name = self.model_name
        request.inputs[self.model_config[self.model]['in_key']].CopyFrom(
            make_tensor_proto(my_image, shape=(my_image.shape))
        )
        # start = time.time()           # Used for Bencharking
        result = stub.Predict(request, 10.0)
        # end = time.time()             # Used for Bencharking
        result = result.outputs[self.model_config[self.model]['out_key']]
        result = make_ndarray(result)
        # print("FPS: {}".format(1/(end-start)))        # Used for Bencharking
        return result    