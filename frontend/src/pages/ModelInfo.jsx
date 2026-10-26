import { useEffect, useState } from "react";
import ArchitectureDiagram from "../components/ArchitectureDiagram.jsx";
import { getModelInfo } from "../services/api.js";

const pct = (x) => `${(x * 100).toFixed(2)}%`;

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => { getModelInfo().then(setInfo).catch((e) => setError(e.message)); }, []);

  const a = info?.architecture;
  const m = info?.test_metrics;
  const t = info?.training;
  return (
    <>
      <section className="card">
        <h2>Model Information</h2>
        {error && <p className="error-text">{error}</p>}
        <table className="table"><tbody>
          <tr><td>Model</td><td>{a?.model ?? "ResNet-18"}</td></tr>
          <tr><td>Method</td><td>{a?.method ?? "Transfer Learning"}</td></tr>
          <tr><td>Input Size</td><td>{a?.input_size ?? "224 × 224"}</td></tr>
          <tr><td>Deep Features</td><td>{a?.deep_features ?? 512}</td></tr>
          <tr><td>Handcrafted Features</td><td>{a?.handcrafted_features ?? 4} ({(a?.handcrafted_names ?? []).join(", ")})</td></tr>
          <tr><td>Total Fused Features</td><td>{a?.total_features ?? 516}</td></tr>
          <tr><td>Classes</td><td>3 — {(a?.classes ?? ["Normal", "Myopia", "Hyperopia"]).join(", ")}</td></tr>
          <tr><td>Confidence threshold</td><td>{info ? `${(info.confidence_threshold * 100).toFixed(0)}% (below this the result is "Uncertain")` : "—"}</td></tr>
        </tbody></table>
      </section>

      <section className="card">
        <h2>Dataset</h2>
        <p>Train {info?.dataset.train ?? 6300} · Validation {info?.dataset.validation ?? 1350} · Test {info?.dataset.test ?? 1350}
          {" "}(2100 / 450 / 450 images per class). The provided split is preserved.</p>
        <p className="muted small">
          See README "Limitations": many validation/test images share an eye identity with training images, so full-test-set
          accuracy is optimistic. A leakage-aware score is reported below when available.
        </p>
      </section>

      <section className="card">
        <h2>Measured Results</h2>
        {!m ? (
          <p className="muted">No evaluation found yet. Run <code>python training/train.py</code> — results appear here after training.</p>
        ) : (
          <>
            <p><strong>Test accuracy (all {m.num_images} test images):</strong> {pct(m.accuracy)} &nbsp;·&nbsp; Macro F1 {m.macro_f1.toFixed(4)} &nbsp;·&nbsp; Weighted F1 {m.weighted_f1.toFixed(4)}</p>
            {m.clean_subset_no_train_near_duplicate && (
              <p><strong>Leakage-aware test accuracy ({m.clean_subset_no_train_near_duplicate.num_images} test images with no near-copy in train):</strong>{" "}
                {pct(m.clean_subset_no_train_near_duplicate.accuracy)}</p>
            )}
            <table className="table"><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
              <tbody>{Object.entries(m.per_class).map(([k, v]) => (
                <tr key={k}><td>{k}</td><td>{v.precision.toFixed(3)}</td><td>{v.recall.toFixed(3)}</td><td>{v.f1.toFixed(3)}</td><td>{v.support}</td></tr>
              ))}</tbody></table>
            {t && <p className="muted small">Trained {t.epochs_run} epochs on {t.device}; best epoch {t.best_epoch}.</p>}
          </>
        )}
      </section>

      <section className="card"><h2>Architecture</h2><ArchitectureDiagram /></section>
    </>
  );
}
