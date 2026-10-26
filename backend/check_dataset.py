"""Verify retrained model predictions on test samples."""
import sys
sys.path.insert(0, 'backend')
import config
from prediction.predict import ScreeningPredictor

predictor = ScreeningPredictor.from_disk()

correct = 0
total = 0

for cls_idx, cls_name in enumerate(config.CLASS_NAMES):
    test_dir = config.DATASET_DIR / "test" / cls_name
    images = sorted([f for f in test_dir.iterdir() if f.suffix.lower() in config.IMAGE_EXTENSIONS])[:10]
    for img_path in images:
        data = img_path.read_bytes()
        try:
            result = predictor.predict(img_path.name, data)
            pred = result['top_class']
            conf = result['confidence']
            is_correct = pred.lower() == cls_name
            correct += int(is_correct)
            total += 1
            mark = "OK" if is_correct else "WRONG"
            print(f"[{mark:5s}] True={cls_name:10s} Pred={pred:10s} Conf={conf:5.1f}% Spectacles={result['spectacles_verdict']}")
        except Exception as e:
            total += 1
            print(f"[ERROR] True={cls_name:10s} Error: {e}")

print(f"\nSample Accuracy: {correct}/{total} = {correct/max(total,1)*100:.1f}%")
