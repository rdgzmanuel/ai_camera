from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11m.pt")  # or yolov11m.pt, etc.

    model.train(
        data="football.yaml",
        epochs=2,
        imgsz=640,
        batch=16,
        lr0=1e-3,
        optimizer="Adam",
        freeze=[0],
        name="yolov11_1"
    )

if __name__ == "__main__":
    main()
