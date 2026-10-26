const STEPS = [
  "Capture only one eye.",
  "Keep the eye fully open.",
  "Look directly at the camera.",
  "Keep the camera at approximately eye level.",
  "Keep the eye centered.",
  "Avoid excessive blinking during capture.",
  "Avoid sunglasses.",
  "Avoid tinted glasses.",
  "Avoid heavy reflections on the eye.",
  "Use sufficient lighting.",
  "Do not use extremely bright flash directly into the eye.",
  "Avoid very dark rooms.",
  "Keep the camera stable.",
  "Keep the eye approximately 20–30 cm from the camera.",
  "Ensure the iris and pupil are clearly visible.",
  "Do not heavily crop or edit the image.",
  "Avoid motion blur.",
  "Avoid extreme shadows.",
  "Remove anything blocking the eye.",
];
const GOOD = ["Single eye", "Clear iris", "Clear pupil", "Good lighting", "Eye centered", "No motion blur"];
const BAD = ["Blurred eye", "Very dark image", "Strong reflection", "Eye partially closed", "Multiple people", "Sunglasses", "Heavy obstruction"];

function EyeIllustration({ kind }) {
  const isGood = kind === "good";
  const imgSrc = isGood ? "/good_eye.jpg?v=2" : "/bad_eye.jpg?v=2";
  const altText = isGood ? "Example of a clear, centered good eye photo" : "Example of a blurry, rejected eye photo";

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        maxWidth: 240,
        height: 130,
        margin: "0 auto 10px",
        borderRadius: 10,
        overflow: "hidden",
        border: isGood ? "2px solid #10b981" : "2px solid #ef4444",
        boxShadow: isGood ? "0 4px 14px rgba(16,185,129,0.25)" : "0 4px 14px rgba(239,68,68,0.25)",
      }}
    >
      <img
        src={imgSrc}
        alt={altText}
        style={{
          width: "100%",
          height: "100%",
          objectFit: "cover",
          display: "block",
        }}
      />
      {!isGood && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: "rgba(239,68,68,0.2)",
            display: "grid",
            placeItems: "center",
          }}
        >
          <span style={{ fontSize: "2.6rem", color: "#ef4444", fontWeight: "900", textShadow: "0 2px 10px rgba(0,0,0,0.9)" }}>
            ✕
          </span>
        </div>
      )}
    </div>
  );
}

export default function InstructionCard() {
  return (
    <section className="card instructions">
      <h2>How to Capture a Good Eye Image</h2>
      <ol className="steps-list">
        {STEPS.map((s) => <li key={s}>{s}</li>)}
      </ol>
      <div className="examples">
        <div className="example good">
          <EyeIllustration kind="good" />
          <h3>✅ Good image</h3>
          <ul>{GOOD.map((g) => <li key={g}>{g}</li>)}</ul>
        </div>
        <div className="example bad">
          <EyeIllustration kind="bad" />
          <h3>❌ Bad image</h3>
          <ul>{BAD.map((b) => <li key={b}>{b}</li>)}</ul>
        </div>
      </div>
    </section>
  );
}
