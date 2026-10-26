import { useEffect, useRef, useState, useCallback } from "react";
import InstructionCard from "../components/InstructionCard.jsx";
import CameraCapture from "../components/CameraCapture.jsx";
import ResultCard from "../components/ResultCard.jsx";
import ProbabilityBars from "../components/ProbabilityBars.jsx";
import FeaturePanel from "../components/FeaturePanel.jsx";
import Disclaimer from "../components/Disclaimer.jsx";
import { predictImage, checkHealth } from "../services/api.js";

const MAX_MB = 8;
const ALLOWED_TYPES = ["image/jpeg", "image/png"];
const ALLOWED_EXTS = /\.(jpe?g|png)$/i;

const STEP_LABELS = [
  "Select Eye Image(s)",
  "Review Image Quality",
  "AI Screening",
  "Spectacles Screening Verdict",
];

/** Validate a file: returns { ok, reason } */
function validateFile(file) {
  if (!file) return { ok: false, reason: "No file selected." };
  const typeOk = ALLOWED_TYPES.includes(file.type) || ALLOWED_EXTS.test(file.name);
  if (!typeOk)
    return { ok: false, reason: `Invalid type "${file.type || file.name}". Only JPG, JPEG or PNG accepted.` };
  if (file.size > MAX_MB * 1024 * 1024)
    return { ok: false, reason: `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_MB} MB.` };
  if (file.size < 1024)
    return { ok: false, reason: "File is too small to be a valid eye image." };
  return { ok: true, reason: null };
}

/** Inspect image to detect blur, lack of clarity, poor lighting, diagrams, and non-eye graphics */
function inspectEyeImage(file, callback) {
  const basic = validateFile(file);
  if (!basic.ok) {
    callback({ ok: false, reason: basic.reason });
    return;
  }
  const img = new Image();
  const url = URL.createObjectURL(file);
  img.onload = () => {
    URL.revokeObjectURL(url);
    try {
      const canvas = document.createElement("canvas");
      // Scale down to max 240px side for fast and accurate pixel processing
      const maxSide = 240;
      const scale = Math.min(1.0, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.max(20, Math.round(img.naturalWidth * scale));
      const h = Math.max(20, Math.round(img.naturalHeight * scale));
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, w, h);
      const data = ctx.getImageData(0, 0, w, h).data;
      const total = w * h;

      let whitePixels = 0;
      let sumGray = 0;
      let sumSqGray = 0;
      const gray = new Float32Array(total);

      for (let i = 0; i < total; i++) {
        const idx = i * 4;
        const r = data[idx], g = data[idx + 1], b = data[idx + 2];
        if (r > 225 && g > 225 && b > 225) {
          whitePixels++;
        }
        const grayVal = 0.299 * r + 0.587 * g + 0.114 * b;
        gray[i] = grayVal;
        sumGray += grayVal;
        sumSqGray += grayVal * grayVal;
      }

      // 1. Diagram / Document check
      const whiteRatio = whitePixels / total;
      if (whiteRatio > 0.35) {
        callback({
          ok: false,
          reason: `Image rejected: Document, diagram, or graphic detected (${(whiteRatio * 100).toFixed(0)}% white background). Only actual eye photos are accepted.`,
        });
        return;
      }

      // 2. Brightness & Contrast check
      const meanBrightness = sumGray / total;
      const contrast = Math.sqrt(Math.max(0, (sumSqGray / total) - (meanBrightness * meanBrightness)));

      if (meanBrightness < 30) {
        callback({
          ok: false,
          reason: "Image rejected: The image is too dark to distinguish eye features. Please capture with adequate lighting.",
        });
        return;
      }

      if (meanBrightness > 245) {
        callback({
          ok: false,
          reason: "Image rejected: The image is overexposed or has excessive glare. Avoid direct flash into the eye.",
        });
        return;
      }

      if (contrast < 18) {
        callback({
          ok: false,
          reason: "Image rejected: Low contrast. The iris and pupil are not clearly visible. Ensure proper lighting.",
        });
        return;
      }

      // 3. Sharpness / Clarity check (Laplacian variance)
      let lapSum = 0;
      let lapSumSq = 0;
      let lapCount = 0;
      for (let y = 1; y < h - 1; y++) {
        const row = y * w;
        const rowPrev = (y - 1) * w;
        const rowNext = (y + 1) * w;
        for (let x = 1; x < w - 1; x++) {
          const center = gray[row + x];
          const lap = gray[rowPrev + x] + gray[rowNext + x] + gray[row + x - 1] + gray[row + x + 1] - 4 * center;
          lapSum += lap;
          lapSumSq += lap * lap;
          lapCount++;
        }
      }
      const lapMean = lapSum / lapCount;
      const sharpness = (lapSumSq / lapCount) - (lapMean * lapMean);

      // Rejects blurry, out-of-focus, or motion-blurred inputs
      if (sharpness < 25.0) {
        callback({
          ok: false,
          reason: `Image rejected: The uploaded eye image is not clear or out of focus (sharpness score ${sharpness.toFixed(1)} is below required clarity). Please upload a sharp, in-focus close-up eye photo.`,
        });
        return;
      }

      callback({ ok: true, reason: null });
    } catch {
      callback({ ok: true, reason: null });
    }
  };
  img.onerror = () => {
    URL.revokeObjectURL(url);
    callback({ ok: false, reason: "Unable to read image." });
  };
  img.src = url;
}

/** One eye upload slot */
function EyeSlot({ label, emoji, eye, onFile, onCamera, onRemove }) {
  const ref = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }, [onFile]);

  return (
    <div
      className={`eye-slot ${eye?.file ? "has-image" : ""} ${eye?.error ? "error-slot" : ""}`}
      onClick={() => !eye?.file && ref.current?.click()}
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
      role="button"
      aria-label={`Upload ${label}`}
    >
      <input
        ref={ref}
        type="file"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        hidden
        id={`file-input-${label.toLowerCase().replace(/\s/g, "-")}`}
        onChange={(e) => { onFile(e.target.files[0]); e.target.value = ""; }}
      />

      {eye?.file ? (
        <>
          <img className="eye-preview" src={eye.url} alt={`${label} preview`} />
          <div className="eye-slot-label">{emoji} {label}</div>
          {eye.valid ? (
            <span className="validation-badge valid">✓ Valid eye image</span>
          ) : (
            <span className="validation-badge invalid">✗ {eye.error}</span>
          )}
          <div className="eye-slot-actions">
            <button className="btn small danger" onClick={(e) => { e.stopPropagation(); onRemove(); }}>
              🗑 Remove
            </button>
            <button className="btn small" onClick={(e) => { e.stopPropagation(); ref.current?.click(); }}>
              ↩ Change
            </button>
            <button className="btn small" onClick={(e) => { e.stopPropagation(); onCamera(); }}>
              📷 Retake
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="eye-slot-icon">{emoji}</div>
          <div className="eye-slot-label">{label}</div>
          <div className="eye-slot-sub">Click or drag & drop<br />JPG / PNG · max {MAX_MB} MB</div>
          <div className="eye-slot-actions" style={{ display: "flex", gap: "8px", justifyContent: "center", marginTop: "10px" }}>
            <button className="btn primary" onClick={(e) => { e.stopPropagation(); ref.current?.click(); }}>
              ⬆ Upload
            </button>
            <button className="btn" onClick={(e) => { e.stopPropagation(); onCamera(); }}>
              📷 Camera
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default function Screen({ intent }) {
  // Single eye: { file, url, valid, error } | null
  const [eye, setEye] = useState(null);
  const [showCamera, setShowCamera] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking"); // "ok" | "error" | "checking"

  const checkBackend = useCallback(() => {
    checkHealth()
      .then((r) => setBackendStatus(r?.status === "ok" ? "ok" : "error"))
      .catch(() => setBackendStatus("error"));
  }, []);

  // Check backend on mount and poll periodically
  useEffect(() => {
    checkBackend();
    const interval = setInterval(checkBackend, 3500);
    return () => clearInterval(interval);
  }, [checkBackend]);

  const eyeUrlRef = useRef(null);

  useEffect(() => {
    eyeUrlRef.current = eye?.url;
  }, [eye?.url]);

  // Revoke object URLs strictly on unmount
  useEffect(() => () => {
    if (eyeUrlRef.current) URL.revokeObjectURL(eyeUrlRef.current);
  }, []);

  const step = result ? 4 : loading ? 3 : eye?.valid ? 2 : 1;

  const handleEyeFile = (file) => {
    if (!file) return;
    if (eye?.url) URL.revokeObjectURL(eye.url);
    const tempUrl = URL.createObjectURL(file);
    setEye({ file, url: tempUrl, valid: false, error: "Checking image quality..." });
    setResult(null);
    setError(null);

    inspectEyeImage(file, ({ ok, reason }) => {
      setEye({ file, url: tempUrl, valid: ok, error: reason });
    });
  };

  const removeEye = () => {
    if (eye?.url) URL.revokeObjectURL(eye.url);
    setEye(null);
    setResult(null);
    setError(null);
  };

  const analyze = async () => {
    if (!eye?.valid) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictImage(eye.file);
      setResult(res);
    } catch (e) {
      setError({
        message:
          e.status === 422
            ? "Unable to analyze this image. Please upload a clear, close-up eye photo (JPG/PNG)."
            : e.network
            ? "Cannot reach the backend. Please start the server: uvicorn main:app --reload"
            : e.message,
        reasons: e.reasons || [],
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Backend status banner */}
      <BackendBanner status={backendStatus} onRetry={checkBackend} />

      <ol className="stepper">
        {STEP_LABELS.map((l, i) => (
          <li key={l} className={step === i + 1 ? "current" : step > i + 1 ? "done" : ""}>
            <span>{i + 1}</span>{l}
          </li>
        ))}
      </ol>

      <Disclaimer />
      <InstructionCard />

      {/* Camera Capture Modal / Section */}
      {showCamera && (
        <section className="card" style={{ marginBottom: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
            <h3 style={{ margin: 0 }}>📷 Capture Eye Image</h3>
            <button className="btn small ghost" onClick={() => setShowCamera(false)}>✖ Close Camera</button>
          </div>
          <CameraCapture
            onUse={(file) => {
              handleEyeFile(file);
              setShowCamera(false);
            }}
            onCancel={() => setShowCamera(false)}
          />
        </section>
      )}

      {/* Step 1: Upload or capture eye image */}
      {!result && !loading && (
        <section className="card">
          <h2>Step 1 — Eye Image Selection</h2>
          <p className="muted small" style={{ marginBottom: 16 }}>
            Upload or capture a single eye image for screening. Only JPG/JPEG/PNG accepted (max {MAX_MB} MB).
          </p>

          <div className="single-eye-wrap" style={{ maxWidth: 540, margin: "0 auto" }}>
            <EyeSlot
              label="Eye Image"
              emoji="👁️"
              eye={eye}
              onFile={handleEyeFile}
              onCamera={() => setShowCamera(true)}
              onRemove={removeEye}
            />
          </div>

          {/* Validation errors */}
          {eye && !eye.valid && (
            <div className="card error-card" style={{ maxWidth: 540, margin: "14px auto 0" }} role="alert">
              <strong>⚠ Image Issue:</strong> {eye.error}
            </div>
          )}

          {eye?.valid && (
            <div className="btn-row center" style={{ marginTop: 24 }}>
              <button className="btn primary big" onClick={analyze} disabled={loading}>
                🔍 Analyze Eye Image
              </button>
            </div>
          )}

          {!eye && (
            <p className="muted small" style={{ textAlign: "center", marginTop: 14 }}>
              ℹ Upload or take a photo of one eye to start analysis.
            </p>
          )}
        </section>
      )}

      {/* Error from analysis */}
      {error && (
        <section className="card error-card" role="alert">
          <strong>❌ {error.message}</strong>
          {error.reasons.length > 0 && (
            <ul style={{ marginTop: 8 }}>
              {error.reasons.map((r) => <li key={r}>{r}</li>)}
            </ul>
          )}
        </section>
      )}

      {/* Loading */}
      {loading && (
        <section className="card loading-section">
          <div className="spinner" />
          <p>Analyzing eye image… extracting 516 features with ResNet-18</p>
        </section>
      )}

      {/* Results */}
      {result && (
        <>
          <h2 className="section-title">Step 4 — AI Screening Result</h2>
          <ResultCard result={result} eye={eye} />

          <div className="grid2" style={{ marginTop: 20 }}>
            <ProbabilityBars probabilities={result.probabilities} />
            <FeaturePanel result={result} />
          </div>

          {result.warnings?.length > 0 && (
            <section className="card warn-card" style={{ marginTop: 16 }}>
              <strong>Notes & Observations</strong>
              <ul>{result.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
            </section>
          )}

          <div className="btn-row center" style={{ marginTop: 24 }}>
            <button className="btn big" onClick={() => {
              removeEye();
              setResult(null);
              setError(null);
            }}>
              🔄 Screen Another Eye
            </button>
          </div>
        </>
      )}
    </>
  );
}

/* ── Backend status banner component ── */
function BackendBanner({ status, onRetry }) {
  const map = {
    checking: { cls: "checking", dot: "checking", msg: "Checking backend connection…" },
    ok: { cls: "ok", dot: "ok", msg: "Backend connected — uvicorn running at http://localhost:8000" },
    error: { cls: "error", dot: "error", msg: "Cannot reach backend. Run: .venv312\\Scripts\\activate → cd backend → uvicorn main:app --reload" },
  };
  const s = map[status] || map.checking;
  return (
    <div className={`backend-banner ${s.cls}`} style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <div>
        <span className={`status-dot ${s.dot}`} />
        {s.msg}
      </div>
      {status === "error" && onRetry && (
        <button
          className="btn small"
          onClick={onRetry}
          style={{ padding: "4px 10px", fontSize: "0.82rem", marginLeft: 12, cursor: "pointer" }}
        >
          🔄 Re-check
        </button>
      )}
    </div>
  );
}

