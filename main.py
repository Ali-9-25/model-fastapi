import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from fastapi import Request
import skimage as ski
from PreProcessing import PreProcessing
from ExtractFeatures import extract_features
import time
app = FastAPI()


model = pickle.load(open("svm_model.pkl", "rb"))
# img
# {time :  result: }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("image is received")
        image = ski.io.imread(file.file)
        print("image is read ", image)

        start_time = time.time()
        print("start_time ", start_time)
        image_processed = PreProcessing(image)

        print("img_processed", image_processed)
        features = extract_features(image_processed)

        print("features before reshape", features.shape)
        features = features.reshape(1, -1)

        print("features are extracted", features.shape)

        result = model.predict(features[:, 1:])
        print("result is predicted", result)

        end_time = time.time()
        print("end time ", end_time)
        time_taken = end_time - start_time

        print("time taken to predict the image is ", end_time - start_time)
        print("result is ", result)

        return JSONResponse(content={"result": str(result[0]), "time": time_taken}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
