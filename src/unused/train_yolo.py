from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11n.pt")

    model.train(
        data="roboflow.yaml",
        epochs=200,
        patience=50,
        imgsz=1088,
        batch=16,
        lr0=1e-3,
        optimizer="Adam",
        close_mosaic=5,
        rect=True,
        name="yolov11n_rf_1088_200"
    )

if __name__ == "__main__":
    main()
