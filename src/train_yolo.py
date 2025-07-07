from ultralytics import YOLO

def main():
    model = YOLO("models/yolo11n.pt")

    model.train(
        data="football.yaml",
        epochs=400,
        patience=50,
        imgsz=1024,
        batch=16,
        lr0=1e-3,
        optimizer="Adam",
        close_mosaic=5,
        rect=True,
        name="yolov11n_nb_1024_400"
    )

if __name__ == "__main__":
    main()
