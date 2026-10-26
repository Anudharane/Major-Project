import { useState } from "react";

const LABELS = {
  focal_sharpness: ["Focal Sharpness", (v) => v.toFixed(2)],
  meridional_asymmetry: ["Meridional Asymmetry", (v) => v.toFixed(3)],
  pupil_contrast_ratio: ["Pupil Contrast Ratio", (v) => v.toFixed(3)],
  reflex_symmetry: ["Reflex Symmetry", (v) => `${v.toFixed(2)}%`],
};

export default function FeaturePanel({ result }) {
  const [show, setShow] = useState(false);
  const vec = result.feature_vector;
  return (
    <section className="card">
      <h3>Extracted Features</h3>
      <p><strong>ResNet-18 Deep Features:</strong> {result.feature_counts.deep}</p>
      <p><strong>Handcrafted Features:</strong> {result.feature_counts.handcrafted} &nbsp;→&nbsp; fused vector of {result.feature_counts.total}</p>
      <table className="table">
        <tbody>
          {Object.entries(LABELS).map(([key, [label, fmt]]) => (
            <tr key={key}><td>{label}</td><td className="num">{fmt(result.features[key])}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">Handcrafted values are image-based proxy measurements, not clinical photorefraction.</p>
      <button className="btn" onClick={() => setShow(!show)}>{show ? "Hide" : "View"} Feature Vector</button>
      {show && vec && (
        <div className="vector-box">
          <p><strong>4 handcrafted (standardised with the training scaler):</strong></p>
          <code>{vec.handcrafted_scaled.join(", ")}</code>
          <p><strong>512 ResNet-18 deep features:</strong></p>
          <code>{vec.deep.join(", ")}</code>
        </div>
      )}
    </section>
  );
}
