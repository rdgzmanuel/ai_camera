from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11m.pt")

    model.train(
        data="football.yaml",
        epochs=16,
        imgsz=640,
        batch=16,
        lr0=1e-4,
        optimizer="Adam",
        name="yolov11_3"
    )

if __name__ == "__main__":
    main()
