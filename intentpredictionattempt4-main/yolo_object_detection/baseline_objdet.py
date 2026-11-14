"""
This is a baseline deployment script for the YOLOv11 Object-Detection Model that was originally trained/tested in Roboflow 

Purpose:
1. Baseline Documentation
2. Easier inference integration with intent-prediction (cnn-lstm)

Setup instructions:
1. Request access to the team's private Roboflow workspace @ https://app.roboflow.com/dssihrc
2. Follow Roboflow's setup instructions to install dependencies and obtain an API key
3. Store the API key in the DSSIHRC_ROBOFLOW_API_KEY environment variable
"""

from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import os
import cv2 as cv
from pathlib import Path

# Setting up the client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["DSSIHRC_ROBOFLOW_API_KEY"] # setup an environment variable for testing
).configure(InferenceConfiguration(
    api_key_transport="header"
))


# Testing the functionality
script_dir = Path(__file__).resolve().parent # parent directory


"""
Note: Everything past this point is strictly for proof-of-functionality

1. Examines the test inputs in an image subdirectory
2. Makes inferences by calling cloud API service
3. Displays color-coded output using OpenCV (only use-case in this program)


Average net inference time: 10 seconds end-to-end 
"""

COLOR_MAPPING = {"gearbox": (255, 0, 0), "screwdriver": (0, 255, 0), "human_hand": (0, 0, 255)}
LEGEND_ITEMS = [
    ("Gearbox: Blue", COLOR_MAPPING["gearbox"]),
    ("Screwdriver: Green", COLOR_MAPPING["screwdriver"]),
    ("Human Hand: Red", COLOR_MAPPING["human_hand"])
]
images_dir = script_dir / "images"
test_functionality_imgs = os.listdir(images_dir)

for img in test_functionality_imgs:
    # TODO: ONLY section to change if deploying new model
    res = client.run_workflow(
        workspace_name="dssihrc",
        workflow_id="robot-workspace-interaction-2-yolo11m-t1-vrobot-workspace-interaction-2-yolo11m-t1-logic",
        images={
            "image": str(images_dir / img)
        },
        use_cache=True
    )

    pixel_mat = cv.imread(str(images_dir / img))
    for idx, (text, text_color) in enumerate(LEGEND_ITEMS):
        cv.putText(
            pixel_mat,
            text,
            (15, 30 + idx * 30),
            cv.FONT_HERSHEY_COMPLEX,
            0.4,
            text_color,
            2
        )

    for prediction in res[0]["predictions"]["predictions"]:
        color = COLOR_MAPPING[prediction["class"]]

        cv.rectangle(
            img=pixel_mat,
            pt1=(int(prediction["x"] - prediction["width"] / 2), int(prediction["y"] - prediction["height"] / 2)),
            pt2=(int(prediction["x"] + prediction["width"] / 2), int(prediction["y"] + prediction["height"] / 2)),
            color=color,
            thickness=2
        )

    cv.imshow(f"Showcase Test Input: {img}", pixel_mat)

cv.waitKey(0)
cv.destroyAllWindows()
