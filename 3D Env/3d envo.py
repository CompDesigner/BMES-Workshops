import cv2 as cv 
import torch 
import numpy as np
import time 

model_type= "MiDaS_small"
#model_type= "DPT_Large"
#model_type= "DPT_Hybrid"

midas= torch.hub.load("intel-isl/MiDaS",model_type)

device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms= torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

cam = cv.VideoCapture(0)

alpha = 0.8  #Temporal smoothing factor
smoothed = None

while True:
    ret, frame = cam.read()
    start = time.time()

    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    depth = cv.normalize(depth, None, 0, 1, cv.NORM_MINMAX, cv.CV_64F)

    #Apply temporal smoothing
    if smoothed is None:
        smoothed = depth
    else:
        smoothed = alpha * smoothed + (1 - alpha) * depth

    depth = (smoothed * 255).astype(np.uint8)

    #Apply a bilateral filter
    depth = cv.bilateralFilter(depth, d=9, sigmaColor=75, sigmaSpace=75)

    colored = cv.applyColorMap(depth, cv.COLORMAP_HSV)

    end = time.time()
    totalTime = end - start
    fps = 1 / totalTime

    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    cv.putText(frame, f'FPS: {int(fps)}', (20, 70), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv.imshow("Cam", frame)
    cv.imshow("3D Envo", colored)

    if cv.waitKey(5) & 0xFF == 27:
        break

cam.release()
cv.destroyAllWindows()

cv.waitKey(0)
