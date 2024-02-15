import torch 
import random
import detector.utils as utils
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from detector.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from detector.utils.torch_utils import select_device, time_synchronized
from detector.utils.google_utils import gsutil_getsize
from detector_utils import load_onnx_model

def run_yolov7_detection(self):
        # Get the selected model file path and raster layer
        model_file = self.dockwidget.mModelPathLineEdit.text()
        layer = self.get_current_raster_layer()

        print("Model File:", model_file)
        print("Selected Layer:", layer)

        if layer is None:
            QMessageBox.warning(self.dockwidget, "Error", "Please select a valid raster layer.")
            return

        # Load the ONNX model
        onnx_model = load_onnx_model(model_file)

        # Perform YOLOv7 detection for the selected raster layer
        try:
            # Replace this with your YOLOv7 inference logic
            detections = self.run_yolov7_inference(onnx_model, layer)

            # Process and visualize the detections (replace this with your specific logic)
            self.process_and_visualize_detections(layer, detections)

        except Exception as e:
            QMessageBox.critical(self.dockwidget, "Error", f"Error during YOLOv7 inference: {str(e)}")

def run_yolov7_inference(self, model, layer):
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Set up YOLOv7 parameters
        imgsz = self.dockwidget.mInferenceSizeSpinBox.value()
        conf_thres = self.dockwidget.mConfidenceThresSpinBox.value()
        iou_thres = self.dockwidget.mIOUThresSpinBox.value()
        augment = self.dockwidget.mAugmentCheckBox.isChecked()

        # Define device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Read raster layer
        # Note: This is a simple example assuming a single-band raster layer. Adapt as needed for multi-band layers.
        provider = layer.dataProvider()
        _, image_data = provider.readPixels(0, 0, layer.width(), layer.height())
        image = image_data.reshape(layer.height(), layer.width())

        # Preprocess the image (replace this with your preprocessing logic)
        img = torch.from_numpy(image).to(device)
        img = img.float()  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 2:
            img = img.unsqueeze(0).unsqueeze(0)
        elif img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        model(img)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        t3 = time_synchronized()

        # Process detections
        detections = []
        for i, det in enumerate(pred):  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], (layer.width(), layer.height())).round()

                # Add detections to the list
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / torch.tensor([layer.width(), layer.height(), layer.width(), layer.height()])).view(-1).tolist()  # normalized xywh
                    detections.append({'bbox': xyxy, 'confidence': conf.item(), 'class_label': int(cls), 'class_name': names[int(cls)], 'color': colors[int(cls)]})

        return detections
