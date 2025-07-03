from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11n.pt")

    model.train(
        data="football.yaml",
        epochs=300,
        patience=50,
        imgsz=1280,
        batch=16,
        lr0=1e-3,
        optimizer="Adam",
        close_mosaic=5,
        rect=True,
        name="yolov11n_17"
    )

if __name__ == "__main__":
    main()
