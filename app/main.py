from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from utils import predict
from starlette.responses import StreamingResponse, Response
from io import BytesIO
from PIL import Image
app=FastAPI()


@app.get("/")
def home():
	return {"health_check":"OK"} 
	
#@app.post("/remove")
#async def remove(file: UploadFile = File(...)):
#	contents=await file.read()
#	image = Image.open(BytesIO(contents)).convert("RGB")
#	mask=predict(image)
#	return {"mask": mask.tolist()}

@app.post("/remove")
async def remove(file: UploadFile = File(...)):
	contents=await file.read()
	image = Image.open(BytesIO(contents)).convert("RGB")
	mask=predict(image)
	img=Image.fromarray((mask*255).astype('uint8'))
	buffered=BytesIO()
	img.save(buffered,format="JPEG")
	img_str=buffered.getvalue()
	return Response(content=img_str, media_type="image/jpeg")