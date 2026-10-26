const BOXES = [
  "Input Eye Image",
  "Validation + Preprocessing (ROI, 224×224, illumination, CLAHE)",
  "ResNet-18 (ImageNet transfer learning) → 512 deep features",
  "+ 4 handcrafted features → Concatenation (516)",
  "Fully Connected → Softmax (3 classes)",
  "Normal / Myopia / Hyperopia (or Uncertain)",
];

export default function ArchitectureDiagram() {
  return (
    <div className="arch">
      {BOXES.map((b, i) => (
        <div key={b}>
          <div className="arch-box">{b}</div>
          {i < BOXES.length - 1 && <div className="arch-arrow">↓</div>}
        </div>
      ))}
    </div>
  );
}
