import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from fastapi import Request
import skimage as ski
from PreProcessing import PreProcessing
from ExtractFeatures import extract_features, normalize, generate_kernels
import time
import skimage as ski
import numpy as np
from sklearn.preprocessing import StandardScaler


kernels = generate_kernels()


model = pickle.load(open("svm.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


app = FastAPI()
# img
# {time :  result: }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = ski.io.imread(file.file)
        start_time = time.time()
        print("start_time ", start_time)
        image_processed = PreProcessing(image)

        features = extract_features(image_processed, kernels)

        features = features.reshape(1, -1)
        # print("feature : after reshape " , features)

        features = scaler.transform(features)

        result = model.predict(features)
        # print("result is predicted" , result)

        end_time = time.time()
        # print("end time " , end_time)
        time_taken = end_time - start_time

        print("time taken to predict the image is ", end_time - start_time)
        print("result is ", result)

        return JSONResponse(content={"result": str(result[0]), "time": time_taken}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
