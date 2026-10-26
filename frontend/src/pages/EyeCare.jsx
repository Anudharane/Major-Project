import EyeCareGuide from "../components/EyeCareGuide.jsx";

export default function EyeCare({ go }) {
  return (
    <>
      <section className="hero" style={{ padding: "40px 20px 28px" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "5px 14px", borderRadius: 20, background: "rgba(59,130,246,0.15)", border: "1px solid rgba(59,130,246,0.3)", color: "var(--primary2)", fontSize: "0.86rem", fontWeight: 600, marginBottom: 14 }}>
          <span>🛡️</span> Visual Health & Preventive Care
        </div>
        <h1>
          Protective Eye Care <span>Before Needing Spectacles</span>
        </h1>
        <p style={{ maxWidth: 720, margin: "0 auto", fontSize: "1.08rem" }}>
          Simple, evidence-based daily habits, nutritional strategies, and visual ergonomics to protect your natural 20/20 vision and halt early-stage refractive error progression.
        </p>

        <div className="btn-row center" style={{ marginTop: 22 }}>
          <button className="btn primary" onClick={() => go("screen", "upload")}>
            👁 Screen Your Eyes Now
          </button>
          <button className="btn" onClick={() => go("home")}>
            🏠 Back to Home
          </button>
        </div>
      </section>

      {/* 4 Pillars Summary Cards */}
      <section className="grid4" style={{ marginTop: 8, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))", gap: 14 }}>
        <div className="card feature-card" style={{ padding: 18 }}>
          <div className="feature-icon" style={{ fontSize: "1.8rem" }}>🕒</div>
          <h3 style={{ fontSize: "1.05rem" }}>20-20-20 Habit</h3>
          <p className="muted small" style={{ margin: 0 }}>
            Every 20 minutes, gaze 20 feet away for 20 seconds. Resets lens accommodation and eliminates eye strain.
          </p>
        </div>

        <div className="card feature-card" style={{ padding: 18 }}>
          <div className="feature-icon" style={{ fontSize: "1.8rem" }}>☀️</div>
          <h3 style={{ fontSize: "1.05rem" }}>Natural Daylight</h3>
          <p className="muted small" style={{ margin: 0 }}>
            60–90 mins daily outdoors stimulates retinal dopamine release, which scientifically halts eyeball elongation.
          </p>
        </div>

        <div className="card feature-card" style={{ padding: 18 }}>
          <div className="feature-icon" style={{ fontSize: "1.8rem" }}>📱</div>
          <h3 style={{ fontSize: "1.05rem" }}>Distance Rule</h3>
          <p className="muted small" style={{ margin: 0 }}>
            Keep smartphones $\ge 35$–$40$ cm away and monitors at arm's length (50–65 cm). Never read in the dark.
          </p>
        </div>

        <div className="card feature-card" style={{ padding: 18 }}>
          <div className="feature-icon" style={{ fontSize: "1.8rem" }}>💧</div>
          <h3 style={{ fontSize: "1.05rem" }}>Blink Consciously</h3>
          <p className="muted small" style={{ margin: 0 }}>
            Screens cut natural blinking by 60%. Take 10 complete blinks every hour to restore corneal tear film.
          </p>
        </div>
      </section>

      {/* Main Interactive Eye Care Guide */}
      <EyeCareGuide isRequired={false} defaultExpanded={true} />

      {/* Early Warning Signs Card */}
      <section className="card" style={{ marginTop: 24, border: "1px solid rgba(239,68,68,0.25)", background: "rgba(239,68,68,0.03)" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
          <span style={{ fontSize: "1.5rem" }}>⚠️</span>
          <h3 style={{ margin: 0, fontSize: "1.18rem" }}>Early Warning Signs: When to Visit an Optometrist</h3>
        </div>
        <p className="muted small" style={{ marginBottom: 14 }}>
          If you experience any of the following symptoms consistently, do not ignore them. Schedule a comprehensive eye examination:
        </p>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 12 }}>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "12px 14px", borderRadius: 8, border: "1px solid var(--card-border)" }}>
            <strong style={{ color: "var(--ink)", fontSize: "0.95rem" }}>🔍 Squinting to See Distance</strong>
            <p className="muted small" style={{ margin: "4px 0 0" }}>Unconsciously narrowing eyelids to read distant signs or screens indicates early myopia (nearsightedness).</p>
          </div>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "12px 14px", borderRadius: 8, border: "1px solid var(--card-border)" }}>
            <strong style={{ color: "var(--ink)", fontSize: "0.95rem" }}>🤕 Frontal Headaches After Reading</strong>
            <p className="muted small" style={{ margin: "4px 0 0" }}>A dull ache across the forehead or behind the brows usually signals accommodative visual fatigue or hyperopia.</p>
          </div>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "12px 14px", borderRadius: 8, border: "1px solid var(--card-border)" }}>
            <strong style={{ color: "var(--ink)", fontSize: "0.95rem" }}>🌙 Glare & Halos Around Lights</strong>
            <p className="muted small" style={{ margin: "4px 0 0" }}>Difficulty with night driving glare or starbursts around headlights suggests corneal astigmatism or tear film instability.</p>
          </div>
          <div style={{ background: "rgba(255,255,255,0.03)", padding: "12px 14px", borderRadius: 8, border: "1px solid var(--card-border)" }}>
            <strong style={{ color: "var(--ink)", fontSize: "0.95rem" }}>🌵 Burning, Gritty, or Dry Eyes</strong>
            <p className="muted small" style={{ margin: "4px 0 0" }}>Frequent rubbing or sandy sensation means meibomian gland dysfunction and dry eye syndrome from heavy screen use.</p>
          </div>
        </div>

        <div style={{ marginTop: 18, textAlign: "center" }}>
          <button className="btn primary" onClick={() => go("screen", "upload")}>
            👁 Start AI Screening for Your Eyes
          </button>
        </div>
      </section>
    </>
  );
}
