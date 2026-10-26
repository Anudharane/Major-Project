import { useState } from "react";

export default function VisionTests({ go }) {
  const [activeTab, setActiveTab] = useState("acuity");

  // Visual Acuity State
  const acuityLevels = [
    { size: 72, label: "20/100", score: "20/100", text: "E" },
    { size: 52, label: "20/70", score: "20/70", text: "E" },
    { size: 36, label: "20/50", score: "20/50", text: "E" },
    { size: 26, label: "20/40", score: "20/40", text: "E" },
    { size: 18, label: "20/30", score: "20/30", text: "E" },
    { size: 13, label: "20/20 (Normal)", score: "20/20", text: "E" },
  ];
  const directions = ["up", "right", "down", "left"];
  const rotMap = { up: -90, right: 0, down: 90, left: 180 };

  const [acuityIndex, setAcuityIndex] = useState(0);
  const [currentDir, setCurrentDir] = useState("right");
  const [acuityComplete, setAcuityComplete] = useState(false);
  const [acuityPassed, setAcuityPassed] = useState([]);

  const handleDirection = (selected) => {
    const isCorrect = selected === currentDir;
    const nextPassed = [...acuityPassed, isCorrect];
    setAcuityPassed(nextPassed);

    if (acuityIndex + 1 < acuityLevels.length && isCorrect) {
      setAcuityIndex(acuityIndex + 1);
      const nextDir = directions[Math.floor(Math.random() * directions.length)];
      setCurrentDir(nextDir);
    } else {
      setAcuityComplete(true);
    }
  };

  const resetAcuity = () => {
    setAcuityIndex(0);
    setAcuityComplete(false);
    setAcuityPassed([]);
    setCurrentDir(directions[Math.floor(Math.random() * directions.length)]);
  };

  // Astigmatism State
  const [astigmatismAnswer, setAstigmatismAnswer] = useState(null);

  // Amsler Grid State
  const [amslerAnswer, setAmslerAnswer] = useState(null);

  // Color Vision State
  const colorPlates = [
    { num: 12, options: [12, 16, 21, "Nothing"], answer: 12, label: "Plate 1 (Demonstration)" },
    { num: 8, options: [3, 8, 9, "Nothing"], answer: 8, label: "Plate 2 (Red-Green Test)" },
    { num: 29, options: [70, 29, 69, "Nothing"], answer: 29, label: "Plate 3 (Red-Green Test)" },
  ];
  const [colorIndex, setColorIndex] = useState(0);
  const [colorScore, setColorScore] = useState(0);
  const [colorFinished, setColorFinished] = useState(false);

  const handleColorOption = (opt) => {
    const isCorrect = opt === colorPlates[colorIndex].answer;
    if (isCorrect) setColorScore((prev) => prev + 1);

    if (colorIndex + 1 < colorPlates.length) {
      setColorIndex(colorIndex + 1);
    } else {
      setColorFinished(true);
    }
  };

  const resetColor = () => {
    setColorIndex(0);
    setColorScore(0);
    setColorFinished(false);
  };

  return (
    <>
      <section className="hero" style={{ padding: "36px 20px 24px" }}>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            padding: "5px 14px",
            borderRadius: 20,
            background: "rgba(59,130,246,0.15)",
            border: "1px solid rgba(59,130,246,0.3)",
            color: "var(--primary2)",
            fontSize: "0.86rem",
            fontWeight: 600,
            marginBottom: 12,
          }}
        >
          <span>👁️</span> Vision Self-Testing Suite
        </div>
        <h1>
          Interactive <span>Vision Tests & Lens Guide</span>
        </h1>
        <p style={{ maxWidth: 720, margin: "0 auto", fontSize: "1.05rem" }}>
          Quick, interactive clinical screening tools to test visual sharpness, astigmatism, macular health, and color perception from your screen.
        </p>

        {/* Tab Switcher */}
        <div
          className="btn-row center"
          style={{
            marginTop: 24,
            display: "flex",
            flexWrap: "wrap",
            gap: 8,
            justifyContent: "center",
          }}
        >
          <button
            className={`btn ${activeTab === "acuity" ? "primary" : "ghost"}`}
            onClick={() => setActiveTab("acuity")}
          >
            📉 Visual Acuity (E-Chart)
          </button>
          <button
            className={`btn ${activeTab === "astigmatism" ? "primary" : "ghost"}`}
            onClick={() => setActiveTab("astigmatism")}
          >
            🎡 Astigmatism Dial
          </button>
          <button
            className={`btn ${activeTab === "amsler" ? "primary" : "ghost"}`}
            onClick={() => setActiveTab("amsler")}
          >
            🏁 Amsler Macular Grid
          </button>
          <button
            className={`btn ${activeTab === "color" ? "primary" : "ghost"}`}
            onClick={() => setActiveTab("color")}
          >
            🎨 Color Vision Test
          </button>
          <button
            className={`btn ${activeTab === "lenses" ? "primary" : "ghost"}`}
            onClick={() => setActiveTab("lenses")}
          >
            👓 Spectacles & Lens Guide
          </button>
        </div>
      </section>

      {/* ── 1. Visual Acuity (Tumbling E Test) ── */}
      {activeTab === "acuity" && (
        <section className="card" style={{ maxWidth: 780, margin: "0 auto" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 8, marginBottom: 16 }}>
            <div>
              <h2 style={{ margin: 0, fontSize: "1.35rem" }}>Tumbling E Visual Acuity Test</h2>
              <p className="muted small" style={{ margin: "4px 0 0" }}>
                Sit at arm's length (about 1 meter / 3 feet away). Cover one eye at a time and indicate which direction the 'E' fingers point.
              </p>
            </div>
            <span className="badge">Level {acuityIndex + 1} of {acuityLevels.length}</span>
          </div>

          {!acuityComplete ? (
            <div style={{ textAlign: "center", padding: "30px 20px", background: "rgba(255,255,255,0.02)", borderRadius: 12, border: "1px solid var(--card-border)" }}>
              <div
                style={{
                  height: 120,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  marginBottom: 20,
                }}
              >
                <div
                  style={{
                    fontSize: `${acuityLevels[acuityIndex].size}px`,
                    fontWeight: 900,
                    lineHeight: 1,
                    fontFamily: "monospace, sans-serif",
                    color: "var(--ink)",
                    transform: `rotate(${rotMap[currentDir]}deg)`,
                    transition: "all 0.15s ease",
                    userSelect: "none",
                  }}
                >
                  E
                </div>
              </div>

              <div className="small muted" style={{ marginBottom: 18 }}>
                Current Target Size: <strong>{acuityLevels[acuityIndex].label}</strong>
              </div>

              <p style={{ fontWeight: 600, marginBottom: 14 }}>Which direction are the prongs pointing?</p>

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(4, 1fr)",
                  gap: 10,
                  maxWidth: 360,
                  margin: "0 auto",
                }}
              >
                <button className="btn" onClick={() => handleDirection("up")}>⬆ Up</button>
                <button className="btn" onClick={() => handleDirection("right")}>➡ Right</button>
                <button className="btn" onClick={() => handleDirection("down")}>⬇ Down</button>
                <button className="btn" onClick={() => handleDirection("left")}>⬅ Left</button>
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "32px 20px", background: "rgba(16,185,129,0.08)", borderRadius: 12, border: "1px solid rgba(16,185,129,0.3)" }}>
              <div style={{ fontSize: "2.5rem", marginBottom: 10 }}>🎉</div>
              <h3 style={{ fontSize: "1.4rem", margin: 0, color: "var(--ink)" }}>Acuity Screening Completed!</h3>
              <p style={{ fontSize: "1.1rem", margin: "10px 0 16px", color: "var(--primary2)" }}>
                Highest Level Successfully Read: <strong>{acuityLevels[Math.max(0, acuityIndex - 1)].label}</strong>
              </p>
              <p className="muted small" style={{ maxWidth: 520, margin: "0 auto 20px" }}>
                {acuityIndex >= 5
                  ? "Excellent visual acuity at screen distance! Your central resolution appears sharp and normal."
                  : "Reduced screen clarity detected. If you struggle to read smaller lines at arm's length, an optometrist consultation is recommended for prescription lenses."}
              </p>
              <div className="btn-row center">
                <button className="btn primary" onClick={resetAcuity}>🔄 Retest Acuity</button>
                <button className="btn" onClick={() => go("screen", "upload")}>👁 Run Full AI Eye Screening</button>
              </div>
            </div>
          )}
        </section>
      )}

      {/* ── 2. Astigmatism Radial Clock Dial ── */}
      {activeTab === "astigmatism" && (
        <section className="card" style={{ maxWidth: 780, margin: "0 auto" }}>
          <h2 style={{ margin: 0, fontSize: "1.35rem" }}>Astigmatism Clock Dial Test</h2>
          <p className="muted small" style={{ margin: "4px 0 18px" }}>
            Astigmatism occurs when the cornea is curved like an oval football rather than a round sphere, causing lines in certain angles to blur.
          </p>

          <div style={{ textAlign: "center", padding: "24px 16px", background: "#ffffff", borderRadius: 12, marginBottom: 20 }}>
            {/* SVG Astigmatism Radial Dial */}
            <svg width="260" height="260" viewBox="0 0 300 300" style={{ display: "inline-block" }}>
              <circle cx="150" cy="150" r="140" fill="none" stroke="#e2e8f0" strokeWidth="2" />
              {[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165].map((deg) => (
                <line
                  key={deg}
                  x1={150 + 130 * Math.cos((deg * Math.PI) / 180)}
                  y1={150 + 130 * Math.sin((deg * Math.PI) / 180)}
                  x2={150 - 130 * Math.cos((deg * Math.PI) / 180)}
                  y2={150 - 130 * Math.sin((deg * Math.PI) / 180)}
                  stroke="#0f172a"
                  strokeWidth="3.5"
                />
              ))}
              <circle cx="150" cy="150" r="12" fill="#ffffff" stroke="#0f172a" strokeWidth="3" />
            </svg>
          </div>

          <div style={{ background: "rgba(255,255,255,0.03)", padding: 18, borderRadius: 10, border: "1px solid var(--card-border)" }}>
            <p style={{ fontWeight: 600, marginBottom: 12 }}>
              Cover one eye, gaze at the center circle. Do all radiating lines look equally dark and sharp, or do some lines appear bolder, darker, or blurred?
            </p>

            <div className="btn-row" style={{ gap: 10 }}>
              <button
                className={`btn ${astigmatismAnswer === "uniform" ? "primary" : ""}`}
                onClick={() => setAstigmatismAnswer("uniform")}
              >
                ✅ All lines look equally sharp & clear
              </button>
              <button
                className={`btn ${astigmatismAnswer === "uneven" ? "danger" : ""}`}
                onClick={() => setAstigmatismAnswer("uneven")}
              >
                ⚠️ Some lines look darker / thicker than others
              </button>
            </div>

            {astigmatismAnswer === "uniform" && (
              <div style={{ marginTop: 16, padding: "12px 14px", borderRadius: 8, background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)" }}>
                <strong style={{ color: "var(--normal)" }}>Normal Corneal Curvature:</strong> You show no obvious signs of significant astigmatism along this meridian.
              </div>
            )}

            {astigmatismAnswer === "uneven" && (
              <div style={{ marginTop: 16, padding: "12px 14px", borderRadius: 8, background: "rgba(249,115,22,0.12)", border: "1px solid rgba(249,115,22,0.3)" }}>
                <strong style={{ color: "#f97316" }}>Possible Astigmatism Detected:</strong> When certain clock axes look noticeably darker or blurred, it indicates light rays are focusing on two different focal points. An optometrist can verify your Cylinder (CYL) and Axis prescription.
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── 3. Amsler Grid Test ── */}
      {activeTab === "amsler" && (
        <section className="card" style={{ maxWidth: 780, margin: "0 auto" }}>
          <h2 style={{ margin: 0, fontSize: "1.35rem" }}>Amsler Grid Macular Test</h2>
          <p className="muted small" style={{ margin: "4px 0 18px" }}>
            The Amsler grid is used clinically to screen for central retinal distortion, macular degeneration, and fluid beneath the fovea.
          </p>

          <div style={{ textAlign: "center", padding: "24px 16px", background: "#000000", borderRadius: 12, marginBottom: 20 }}>
            {/* SVG Amsler Grid */}
            <svg width="240" height="240" viewBox="0 0 240 240" style={{ display: "inline-block" }}>
              <rect width="240" height="240" fill="#000000" />
              {Array.from({ length: 13 }).map((_, i) => (
                <g key={i}>
                  <line x1={i * 20} y1="0" x2={i * 20} y2="240" stroke="#ffffff" strokeWidth="1" />
                  <line x1="0" y1={i * 20} x2="240" y2={i * 20} stroke="#ffffff" strokeWidth="1" />
                </g>
              ))}
              <circle cx="120" cy="120" r="5" fill="#ffffff" />
            </svg>
          </div>

          <div style={{ background: "rgba(255,255,255,0.03)", padding: 18, borderRadius: 10, border: "1px solid var(--card-border)" }}>
            <p style={{ fontWeight: 600, marginBottom: 12 }}>
              Hold screen at reading distance. Cover one eye and fixate directly on the white center dot. Do any of the lines look wavy, curved, distorted, or missing?
            </p>

            <div className="btn-row" style={{ gap: 10 }}>
              <button
                className={`btn ${amslerAnswer === "straight" ? "primary" : ""}`}
                onClick={() => setAmslerAnswer("straight")}
              >
                ✅ All lines are completely straight & continuous
              </button>
              <button
                className={`btn ${amslerAnswer === "wavy" ? "danger" : ""}`}
                onClick={() => setAmslerAnswer("wavy")}
              >
                ⚠️ Some lines appear wavy, distorted, or have gray patches
              </button>
            </div>

            {amslerAnswer === "straight" && (
              <div style={{ marginTop: 16, padding: "12px 14px", borderRadius: 8, background: "rgba(16,185,129,0.12)", border: "1px solid rgba(16,185,129,0.3)" }}>
                <strong style={{ color: "var(--normal)" }}>Healthy Central Field:</strong> No central visual distortion or metamorphopsia observed.
              </div>
            )}

            {amslerAnswer === "wavy" && (
              <div style={{ marginTop: 16, padding: "12px 14px", borderRadius: 8, background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)" }}>
                <strong style={{ color: "#ef4444" }}>Urgent Retinal Evaluation Recommended:</strong> Wavy or missing lines on an Amsler grid can be an indicator of epiretinal membrane, macular edema, or age-related macular changes. Please schedule an examination with an eye specialist promptly.
              </div>
            )}
          </div>
        </section>
      )}

      {/* ── 4. Color Vision Test ── */}
      {activeTab === "color" && (
        <section className="card" style={{ maxWidth: 780, margin: "0 auto" }}>
          <h2 style={{ margin: 0, fontSize: "1.35rem" }}>Color Vision (Ishihara Style) Check</h2>
          <p className="muted small" style={{ margin: "4px 0 18px" }}>
            Assesses red-green and protan/deutan color discrimination by testing numbers embedded in colored dot matrices.
          </p>

          {!colorFinished ? (
            <div style={{ textAlign: "center", padding: "26px 16px", background: "rgba(255,255,255,0.02)", borderRadius: 12, border: "1px solid var(--card-border)" }}>
              <div className="small muted" style={{ marginBottom: 12 }}>
                {colorPlates[colorIndex].label} — {colorIndex + 1} of {colorPlates.length}
              </div>

              {/* Simulated Colorful Dot Plate Container */}
              <div
                style={{
                  width: 180,
                  height: 180,
                  borderRadius: "50%",
                  background:
                    colorIndex === 0
                      ? "radial-gradient(circle, #f97316 0%, #10b981 70%, #064e3b 100%)"
                      : colorIndex === 1
                      ? "radial-gradient(circle, #ef4444 0%, #84cc16 60%, #15803d 100%)"
                      : "radial-gradient(circle, #eab308 0%, #3b82f6 70%, #1e3a8a 100%)",
                  margin: "0 auto 20px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 0 25px rgba(0,0,0,0.5)",
                }}
              >
                <span
                  style={{
                    fontSize: "3.4rem",
                    fontWeight: 900,
                    color: "#ffffff",
                    letterSpacing: 2,
                    textShadow: "0 2px 8px rgba(0,0,0,0.6)",
                    userSelect: "none",
                  }}
                >
                  {colorPlates[colorIndex].num}
                </span>
              </div>

              <p style={{ fontWeight: 600, marginBottom: 14 }}>What number do you see in the circle above?</p>

              <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
                {colorPlates[colorIndex].options.map((opt) => (
                  <button key={opt} className="btn" onClick={() => handleColorOption(opt)} style={{ minWidth: 70 }}>
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "30px 20px", background: "rgba(16,185,129,0.08)", borderRadius: 12, border: "1px solid rgba(16,185,129,0.3)" }}>
              <div style={{ fontSize: "2.5rem", marginBottom: 10 }}>🎨</div>
              <h3 style={{ fontSize: "1.4rem", margin: 0 }}>Color Discrimination Score: {colorScore} / {colorPlates.length}</h3>
              <p className="muted small" style={{ maxWidth: 480, margin: "10px auto 20px" }}>
                {colorScore === colorPlates.length
                  ? "Normal color perception detected! You have clear differentiation across red, green, and yellow spectrums."
                  : "Some color plates were difficult to distinguish. If you have trouble distinguishing traffic lights, colors, or maps, mention this during your next optometrist checkup."}
              </p>
              <button className="btn primary" onClick={resetColor}>🔄 Retest Color Vision</button>
            </div>
          )}
        </section>
      )}

      {/* ── 5. Spectacles & Lens Guide ── */}
      {activeTab === "lenses" && (
        <section className="card" style={{ maxWidth: 840, margin: "0 auto" }}>
          <h2 style={{ margin: 0, fontSize: "1.35rem" }}>Understanding Spectacles, Lenses & Prescriptions</h2>
          <p className="muted small" style={{ margin: "4px 0 20px" }}>
            A comprehensive reference to make sense of your eye prescription parameters and choose the right optical coatings.
          </p>

          <div className="grid2" style={{ gap: 16 }}>
            {/* Prescription Breakdown */}
            <div style={{ background: "rgba(255,255,255,0.03)", padding: 18, borderRadius: 10, border: "1px solid var(--card-border)" }}>
              <h3 style={{ fontSize: "1.1rem", marginBottom: 12, color: "var(--primary2)" }}>📋 How to Read Your Prescription</h3>
              <ul style={{ paddingLeft: 18, fontSize: "0.92rem", lineHeight: 1.7 }} className="muted">
                <li><strong style={{ color: "var(--ink)" }}>OD vs OS:</strong> <em>OD</em> (Oculus Dexter) = Right eye; <em>OS</em> (Oculus Sinister) = Left eye.</li>
                <li><strong style={{ color: "var(--ink)" }}>SPH (Sphere):</strong> Negative (-) indicates Myopia (nearsightedness); Positive (+) indicates Hyperopia (farsightedness).</li>
                <li><strong style={{ color: "var(--ink)" }}>CYL (Cylinder):</strong> Amount of lens power needed to correct astigmatism.</li>
                <li><strong style={{ color: "var(--ink)" }}>AXIS (0°–180°):</strong> The precise orientation meridian of astigmatism correction.</li>
                <li><strong style={{ color: "var(--ink)" }}>ADD:</strong> Additional magnifying power needed for reading (Presbyopia).</li>
              </ul>
            </div>

            {/* Lens Coatings Guide */}
            <div style={{ background: "rgba(255,255,255,0.03)", padding: 18, borderRadius: 10, border: "1px solid var(--card-border)" }}>
              <h3 style={{ fontSize: "1.1rem", marginBottom: 12, color: "var(--primary2)" }}>🛡️ Optical Lens Coatings</h3>
              <ul style={{ paddingLeft: 18, fontSize: "0.92rem", lineHeight: 1.7 }} className="muted">
                <li><strong style={{ color: "var(--ink)" }}>Anti-Reflective (AR):</strong> Eliminates halos, night driving glare, and reflections so 99.5% of light reaches the retina.</li>
                <li><strong style={{ color: "var(--ink)" }}>Blue-Light Filter:</strong> Selectively blocks 415–455nm high-energy visible light from monitors and smartphones.</li>
                <li><strong style={{ color: "var(--ink)" }}>Photochromic (Transition):</strong> Lenses darken automatically when exposed to outdoor UV sunlight.</li>
                <li><strong style={{ color: "var(--ink)" }}>High-Index (1.67 / 1.74):</strong> Ultrathin, lightweight lenses for stronger prescriptions to avoid thick edges.</li>
              </ul>
            </div>
          </div>

          <div style={{ marginTop: 22, textAlign: "center" }}>
            <button className="btn primary" onClick={() => go("screen", "upload")}>
              👁 Start AI Vision Screening
            </button>
          </div>
        </section>
      )}
    </>
  );
}
