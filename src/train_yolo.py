from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11s.pt")

    model.train(
        data="football.yaml",
        epochs=16,
        patience=5,
        imgsz=640,
        batch=16,
        lr0=1e-4,
        optimizer="Adam",
        close_mosaic=5,
        name="yolov11s_1"
    )

if __name__ == "__main__":
    main()
