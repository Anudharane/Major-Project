import Disclaimer from "../components/Disclaimer.jsx";

export default function Home({ go }) {
  return (
    <>
      <section className="hero">
        <h1>AI <span>Vision Screening</span></h1>
        <p>AI-Based Preliminary Spectacle Requirement Screening using ResNet-18 Transfer Learning</p>
        <div className="hero-badges">
          <span className="badge">React.js + Vite</span>
          <span className="badge purple">FastAPI + Uvicorn</span>
          <span className="badge green">ResNet-18 Deep Learning</span>
          <span className="badge">NumPy + Pillow</span>
        </div>
        <div className="btn-row center" style={{ marginTop: 24 }}>
          <button className="btn primary big" onClick={() => go("screen", "upload")}>
            👁 Start Eye Screening
          </button>
          <button className="btn big" onClick={() => go("tests")}>
            🧪 Vision Self-Tests
          </button>
        </div>
      </section>

      <section className="grid3" style={{ marginTop: 8 }}>
        <div className="card feature-card">
          <div className="feature-icon">📷</div>
          <h3>Single Eye Upload</h3>
          <p>Upload or capture a single close-up eye photo. The AI analyzes the eye image to screen for spectacle requirement.</p>
        </div>
        <div className="card feature-card">
          <div className="feature-icon">🔍</div>
          <h3>Deep AI Analysis</h3>
          <p>ResNet-18 transfer learning extracts <strong>512 deep features</strong> combined with 4 handcrafted ocular features = 516 total for classification.</p>
        </div>
        <div className="card feature-card">
          <div className="feature-icon">👓</div>
          <h3>Spectacle Requirement</h3>
          <p>Predicts whether <strong>Spectacles are Required or Not Required</strong> with a confidence score and refractive status analysis.</p>
        </div>
      </section>

      <div className="card" style={{ marginTop: 8 }}>
        <h3>🛡 Image Validation</h3>
        <p className="muted" style={{ marginBottom: 12 }}>Only proper eye images are accepted. The system rejects:</p>
        <div className="grid2">
          <div className="example good">
            <div style={{ width: "100%", height: 110, borderRadius: 8, overflow: "hidden", marginBottom: 10, border: "2px solid #10b981" }}>
              <img src="/good_eye.jpg" alt="Clear, valid eye" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            </div>
            <h3>✅ Accepted</h3>
            <ul>
              <li>Clear, close-up eye photo</li>
              <li>JPG / JPEG / PNG format</li>
              <li>Good lighting, in focus</li>
              <li>Single eye visible</li>
              <li>Under {8} MB file size</li>
            </ul>
          </div>
          <div className="example bad">
            <div style={{ position: "relative", width: "100%", height: 110, borderRadius: 8, overflow: "hidden", marginBottom: 10, border: "2px solid #ef4444" }}>
              <img src="/bad_eye.jpg?v=2" alt="Blurry, rejected eye" style={{ width: "100%", height: "100%", objectFit: "cover" }} />
              <div style={{ position: "absolute", inset: 0, background: "rgba(239,68,68,0.25)", display: "grid", placeItems: "center" }}>
                <span style={{ fontSize: "2.4rem", color: "#ef4444", fontWeight: "900", textShadow: "0 2px 8px rgba(0,0,0,0.8)" }}>✕</span>
              </div>
            </div>
            <h3>❌ Rejected</h3>
            <ul>
              <li>Non-eye images (faces, objects)</li>
              <li>Blurry or very dark images</li>
              <li>GIF, BMP, HEIC, PDF formats</li>
              <li>Overexposed or washed-out photos</li>
              <li>Files over 8 MB</li>
            </ul>
          </div>
        </div>
      </div>

      <Disclaimer />
    </>
  );
}
