export default function ProbabilityBars({ probabilities }) {
  const normal = probabilities.Normal ?? probabilities.normal ?? 0;
  const myopia = probabilities.Myopia ?? probabilities.myopia ?? 0;
  const hyperopia = probabilities.Hyperopia ?? probabilities.hyperopia ?? 0;
  const spectacles = parseFloat((myopia + hyperopia).toFixed(1));

  return (
    <section className="card">
      <h3 style={{ margin: "0 0 14px 0" }}>Spectacles Requirement Probability</h3>

      {/* Top Labels on One Line */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "8px",
          marginBottom: "10px",
          fontWeight: 600,
        }}
      >
        <div style={{ color: "#f87171", display: "flex", alignItems: "center", gap: "6px" }}>
          <span>👓 Spectacles Required</span>
          <strong style={{ fontSize: "1.1rem" }}>{spectacles.toFixed(1)}%</strong>
        </div>

        <div style={{ color: "#34d399", display: "flex", alignItems: "center", gap: "6px" }}>
          <span>🟢 Spectacles Not Required</span>
          <strong style={{ fontSize: "1.1rem" }}>{normal.toFixed(1)}%</strong>
        </div>
      </div>

      {/* Single Combined Multi-segment Bar (Red + Green in one) */}
      <div
        style={{
          width: "100%",
          height: "22px",
          background: "rgba(255, 255, 255, 0.08)",
          borderRadius: "12px",
          overflow: "hidden",
          display: "flex",
          boxShadow: "inset 0 1px 3px rgba(0, 0, 0, 0.5)",
        }}
      >
        {spectacles > 0 && (
          <div
            style={{
              width: `${spectacles}%`,
              background: "linear-gradient(90deg, #f97316, #ef4444)",
              height: "100%",
              transition: "width 0.6s ease",
            }}
            title={`Spectacles Required: ${spectacles.toFixed(1)}%`}
          />
        )}
        {normal > 0 && (
          <div
            style={{
              width: `${normal}%`,
              background: "linear-gradient(90deg, #10b981, #059669)",
              height: "100%",
              transition: "width 0.6s ease",
            }}
            title={`Spectacles Not Required: ${normal.toFixed(1)}%`}
          />
        )}
      </div>

      {/* Ratio Guide */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          marginTop: "8px",
          fontSize: "0.8rem",
          color: "var(--muted)",
        }}
      >
        <span>Red: Spectacles Required</span>
        <span>Green: Spectacles Not Required</span>
      </div>
    </section>
  );
}
