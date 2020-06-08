import cv2
import numpy as np
import redis
import time
import pickle
from PIL import Image
import io

cap = cv2.VideoCapture(0)

HOST="35.225.36.129"
PORT=6379

SCALE_DOWN = 0.2

redis_client = redis.Redis(host=HOST, port=PORT)
print(redis_client.keys())

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=SCALE_DOWN, fy=SCALE_DOWN, interpolation=cv2.INTER_AREA)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    _time = time.time()
    img_pil = Image.fromarray(frame)
    imgByteArr = io.BytesIO()
    img_pil.save(imgByteArr, format='JPEG')
    cv2.imshow('Input', frame)
    redis_client.set("I", imgByteArr.getvalue())
    print(time.time() - _time)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
