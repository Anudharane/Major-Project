import { useState } from "react";

export default function EyeCareGuide({ isRequired = false, defaultExpanded = true }) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  const [activeTab, setActiveTab] = useState("daily");

  const categories = {
    daily: {
      title: "Daily Screen Habits",
      icon: "💻",
      tips: [
        {
          heading: "Follow the 20-20-20 Rule",
          desc: "Every 20 minutes spent on a screen, gaze at an object at least 20 feet (6 meters) away for 20 seconds. This relaxes the ciliary eye muscles and prevents digital accommodative spasm.",
          badge: "High Priority",
        },
        {
          heading: "Maintain Safe Viewing Distance",
          desc: "Keep smartphone screens at least 35–40 cm (14–16 inches) away and laptop monitors at 50–65 cm (arm's length). Never hold devices close to your eyes.",
          badge: "Distance",
        },
        {
          heading: "Active Blinking to Prevent Dry Eyes",
          desc: "Reading screens reduces blink rate by over 60%. Consciously take 10 slow, full blinks every hour to spread the natural tear film evenly across the cornea.",
          badge: "Moisture",
        },
        {
          heading: "Never Use Screens in Complete Darkness",
          desc: "A bright display in a dark room creates severe pupillary contrast strain. Always keep soft ambient room lighting on when viewing screens at night.",
          badge: "Lighting",
        },
      ],
    },
    lifestyle: {
      title: "Outdoor & Lifestyle",
      icon: "☀️",
      tips: [
        {
          heading: "Get 60–90 Minutes of Natural Daylight Daily",
          desc: "Sunlight triggers dopamine release in the retina, which scientifically prevents abnormal axial elongation of the eyeball (the biological root cause of myopia/nearsightedness).",
          badge: "Proven Prevention",
        },
        {
          heading: "Optimize Ergonomic Screen Angle",
          desc: "Position your computer screen about 15 to 20 degrees below horizontal eye level. This reduces the exposed ocular surface area, cutting tear evaporation by 40%.",
          badge: "Ergonomics",
        },
        {
          heading: "Quality Sleep & Rest",
          desc: "Aim for 7–8 hours of restorative sleep to allow eye muscles to recover from daily focus fatigue and promote corneal cellular repair.",
          badge: "Recovery",
        },
      ],
    },
    nutrition: {
      title: "Ocular Nutrition",
      icon: "🥗",
      tips: [
        {
          heading: "Lutein & Zeaxanthin (Retinal Filters)",
          desc: "Eat dark leafy greens (spinach, kale, broccoli) and eggs. These natural carotenoids accumulate in the macula to filter harmful blue wavelengths.",
          badge: "Macula Support",
        },
        {
          heading: "Omega-3 Fatty Acids (Tear Film Health)",
          desc: "Include walnuts, flaxseeds, chia seeds, or fatty fish. Omega-3s lubricate meibomian glands, preventing dry, irritated, gritty eyes.",
          badge: "Dry Eye Relief",
        },
        {
          heading: "Vitamin A, C, E & Zinc",
          desc: "Carrots, sweet potatoes, bell peppers, almonds, and citrus fruits support rhodopsin regeneration and cornea clarity.",
          badge: "Antioxidants",
        },
      ],
    },
    exercises: {
      title: "Eye Exercises & Relief",
      icon: "🔄",
      tips: [
        {
          heading: "Near-to-Far Focus Shifts",
          desc: "Hold your thumb 25 cm from your face, focus for 5 seconds, then switch focus to an object across the room for 5 seconds. Repeat 5 times to maintain flexible lens accommodation.",
          badge: "Flexibility",
        },
        {
          heading: "Figure-8 Visual Tracking",
          desc: "Imagine a giant figure 8 on the floor 10 feet away. Slowly trace it with your eyes for 30 seconds clockwise, then reverse. Enhances extraocular muscle coordination.",
          badge: "Tracking",
        },
        {
          heading: "Warm Palming for Tension Relief",
          desc: "Rub your palms together until warm, gently cup them over closed eyes without pressing for 60 seconds. The darkness and warmth soothe fatigued eye muscles.",
          badge: "Relaxation",
        },
      ],
    },
  };

  return (
    <section className="card eye-care-card" style={{ marginTop: 24, border: "1px solid rgba(59,130,246,0.25)" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          cursor: "pointer",
          userSelect: "none",
        }}
        onClick={() => setExpanded(!expanded)}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: "1.7rem" }}>🛡️</span>
          <div>
            <h3 style={{ margin: 0, fontSize: "1.25rem", color: "var(--ink)" }}>
              {isRequired
                ? "Preventive Care & Vision Protection (Halt Further Progression)"
                : "Eye Care Habits to Prevent Needing Spectacles (Protect 20/20 Vision)"}
            </h3>
            <p className="muted small" style={{ margin: "3px 0 0" }}>
              {isRequired
                ? "Essential daily precautions to relieve ocular fatigue and prevent your eye power from worsening."
                : "Evidence-based habits to protect healthy eyes and prevent onset of myopia or digital eye strain."}
            </p>
          </div>
        </div>
        <button
          className="btn small ghost"
          style={{ minWidth: 40, height: 36, fontSize: "1.1rem" }}
          aria-label={expanded ? "Collapse eye care guide" : "Expand eye care guide"}
        >
          {expanded ? "▲" : "▼"}
        </button>
      </div>

      {expanded && (
        <div style={{ marginTop: 20 }}>
          {/* Banner message based on screening status */}
          <div
            style={{
              padding: "12px 16px",
              borderRadius: "10px",
              background: isRequired ? "rgba(249,115,22,0.12)" : "rgba(16,185,129,0.12)",
              border: `1px solid ${isRequired ? "rgba(249,115,22,0.3)" : "rgba(16,185,129,0.3)"}`,
              marginBottom: 16,
              display: "flex",
              alignItems: "center",
              gap: 12,
            }}
          >
            <span style={{ fontSize: "1.4rem" }}>{isRequired ? "👓" : "🌟"}</span>
            <div style={{ fontSize: "0.92rem", color: "var(--ink)" }}>
              {isRequired ? (
                <span>
                  <strong>Early Action Plan:</strong> Even if spectacles are required, practicing these habits prevents rapid increase in refractive error, reduces eye headaches, and safeguards retinal health.
                </span>
              ) : (
                <span>
                  <strong>Preventive Protection:</strong> Your visual acuity is currently normal! Consistent adherence to these 4 pillars prevents digital myopia and keeps your eyesight sharp without corrective lenses.
                </span>
              )}
            </div>
          </div>

          {/* Category Tabs */}
          <div
            style={{
              display: "flex",
              gap: 8,
              flexWrap: "wrap",
              borderBottom: "1px solid var(--card-border)",
              paddingBottom: 12,
              marginBottom: 18,
            }}
          >
            {Object.entries(categories).map(([key, cat]) => (
              <button
                key={key}
                onClick={() => setActiveTab(key)}
                className={`btn small ${activeTab === key ? "primary" : "ghost"}`}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  fontWeight: activeTab === key ? 600 : 400,
                  borderRadius: "20px",
                }}
              >
                <span>{cat.icon}</span>
                <span>{cat.title}</span>
              </button>
            ))}
          </div>

          {/* Tips List */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
              gap: 14,
            }}
          >
            {categories[activeTab].tips.map((tip, idx) => (
              <div
                key={idx}
                style={{
                  background: "rgba(255,255,255,0.03)",
                  border: "1px solid var(--card-border)",
                  borderRadius: "12px",
                  padding: "16px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "space-between",
                  transition: "transform 0.2s, border-color 0.2s",
                }}
              >
                <div>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "flex-start",
                      marginBottom: 8,
                    }}
                  >
                    <h4 style={{ margin: 0, fontSize: "1.02rem", color: "var(--ink)" }}>{tip.heading}</h4>
                    <span
                      style={{
                        fontSize: "0.72rem",
                        padding: "2px 8px",
                        borderRadius: "12px",
                        background: "rgba(59,130,246,0.18)",
                        color: "var(--primary2)",
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                      }}
                    >
                      {tip.badge}
                    </span>
                  </div>
                  <p className="muted small" style={{ lineHeight: 1.55, margin: 0 }}>
                    {tip.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>

          {/* Quick Checklist Footer */}
          <div
            style={{
              marginTop: 18,
              padding: "12px 16px",
              background: "rgba(255,255,255,0.02)",
              borderRadius: "10px",
              border: "1px dashed var(--card-border)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 10,
            }}
          >
            <div className="small muted">
              📌 <em>Did you know?</em> More than 80% of preventable early-stage visual strain can be reversed with daily 20-20-20 breaks and adequate natural daylight exposure.
            </div>
            <span className="small" style={{ color: "var(--primary2)", fontWeight: 500 }}>
              Annual checkup recommended
            </span>
          </div>
        </div>
      )}
    </section>
  );
}
