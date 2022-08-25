from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline.main import Pipeline

class Inputs(BaseModel):
	inputs: list = []

pipeline = Pipeline(data_path="./data", model_path="./models/crf_bilstm")

app = FastAPI()

origins = [
	"http://localhost:3000",
	# add the vercel domain here later
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"]
)

@app.post("/api/predict")
def predict(inputs: Inputs):
	inputs = (inputs.dict())["inputs"]
	results = pipeline.run(inputs)
	return results
