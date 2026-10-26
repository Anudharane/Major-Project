export default function ResultCard({ result, eye }) {
  const probs = result.probabilities || {};
  const normalProb = probs.Normal ?? probs.normal ?? 0;
  const myopiaProb = probs.Myopia ?? probs.myopia ?? 0;
  const hyperopiaProb = probs.Hyperopia ?? probs.hyperopia ?? 0;
  const spectaclesProb = parseFloat((myopiaProb + hyperopiaProb).toFixed(1));

  // Determine binary outcome (Spectacles Required or Not)
  const top = (result.top_class || result.prediction || "").toLowerCase();
  const isRequired = result.spectacles_needed !== undefined
    ? result.spectacles_needed
    : (top !== "normal" || spectaclesProb >= 50);

  const verdictTitle = isRequired ? "SPECTACLES REQUIRED" : "SPECTACLES NOT REQUIRED";
  const verdictEmoji = isRequired ? "👓" : "🟢";
  const verdictConfidence = isRequired
    ? Math.max(spectaclesProb, result.confidence || 0)
    : Math.max(normalProb, result.confidence || 0);

  return (
    <section className={`card result ${isRequired ? "myopia" : "normal"}`}>
      <div className="result-title">AI VISION SCREENING VERDICT</div>

      {eye?.url && (
        <div style={{ margin: "14px auto 6px", width: 84, height: 84, borderRadius: "50%", overflow: "hidden", border: "2px solid rgba(255,255,255,0.25)", boxShadow: "0 4px 12px rgba(0,0,0,0.2)" }}>
          <img src={eye.url} alt="Screened Eye" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        </div>
      )}

      <div className="result-class" style={{ fontSize: "2.3rem", margin: "14px 0", letterSpacing: "0.5px" }}>
        {verdictEmoji} {verdictTitle}
      </div>

      <div className="result-conf" style={{ fontSize: "1.1rem" }}>
        Confidence: <strong>{verdictConfidence.toFixed(1)}%</strong>
      </div>

      <div className="result-msg" style={{ fontSize: "1.15rem", marginTop: 16, fontWeight: 600 }}>
        {isRequired
          ? "👓 Spectacles are required based on this vision screening."
          : "✅ Normal vision detected. Spectacles are not required."}
      </div>

      {result.low_confidence && (
        <div
          style={{
            marginTop: 14,
            padding: "10px 14px",
            borderRadius: 8,
            background: "rgba(245,158,11,0.13)",
            border: "1px solid rgba(245,158,11,0.4)",
            fontSize: "0.9rem",
            display: "flex",
            alignItems: "flex-start",
            gap: 8,
          }}
        >
          <span style={{ fontSize: "1.1rem" }}>⚠️</span>
          <span>
            <strong>Low confidence:</strong> This image differs from the training dataset
            (camera/lighting conditions). The prediction has been conservatively calibrated.
            For reliable results, please visit an optometrist.
          </span>
        </div>
      )}

      <p className="result-note" style={{ marginTop: 14 }}>
        ⚕ Preliminary AI screening only. Please consult a qualified optometrist or eye-care specialist for an official prescription.
      </p>
    </section>
  );
}
