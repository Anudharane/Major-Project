import { useEffect, useRef, useState } from "react";

/** Open camera -> live preview -> Capture (single frame) -> Review -> Use Image / Retake.
 *  Frames are never streamed to the model: only the one captured image is sent later. */
export default function CameraCapture({ onUse, onCancel }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const [facing, setFacing] = useState("environment");
  const [error, setError] = useState("");
  const [shot, setShot] = useState(null); // { file, url }

  const stop = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  };

  useEffect(() => {
    if (shot) return undefined;
    let cancelled = false;
    (async () => {
      setError("");
      if (!navigator.mediaDevices?.getUserMedia) {
        setError("Camera access is not available. Use HTTPS (or localhost), or upload an image instead.");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: facing, width: { ideal: 1280 }, height: { ideal: 960 } },
          audio: false,
        });
        if (cancelled) return stream.getTracks().forEach((t) => t.stop());
        streamRef.current = stream;
        if (videoRef.current) videoRef.current.srcObject = stream;
      } catch (e) {
        setError(
          e.name === "NotAllowedError"
            ? "Camera permission was denied. Allow camera access in the browser, or upload an image instead."
            : "Could not open the camera. Upload an image instead."
        );
      }
    })();
    return () => { cancelled = true; stop(); };
  }, [facing, shot]);

  const capture = () => {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    const c = document.createElement("canvas");
    // Scale down to max 640px for rapid upload and instantaneous inference
    const maxSide = 640;
    const scale = Math.min(1.0, maxSide / Math.max(v.videoWidth, v.videoHeight));
    const targetW = Math.round(v.videoWidth * scale);
    const targetH = Math.round(v.videoHeight * scale);
    c.width = targetW;
    c.height = targetH;
    const ctx = c.getContext("2d");
    ctx.drawImage(v, 0, 0, targetW, targetH);
    c.toBlob((blob) => {
      if (!blob) return;
      stop();
      const file = new File([blob], `capture_${Date.now()}.jpg`, { type: "image/jpeg" });
      setShot({ file, url: URL.createObjectURL(blob) });
    }, "image/jpeg", 0.88);
  };

  const retake = () => { URL.revokeObjectURL(shot.url); setShot(null); };

  return (
    <div className="card camera">
      <h2>Capture From Camera</h2>
      {error && <p className="error-text">{error}</p>}
      {!shot ? (
        <>
          <div className="video-wrap">
            <video ref={videoRef} autoPlay playsInline muted />
            <div className="guide-oval" aria-hidden="true" />
          </div>
          <p className="hint">Center one open eye inside the oval, about 20–30 cm away, with good lighting.</p>
          <div className="btn-row">
            <button className="btn primary" onClick={capture} disabled={!!error}>📸 Capture Image</button>
            <button className="btn" onClick={() => setFacing(facing === "user" ? "environment" : "user")}>🔄 Switch Camera</button>
            <button className="btn ghost" onClick={onCancel}>Cancel</button>
          </div>
        </>
      ) : (
        <>
          <img className="preview-img" src={shot.url} alt="Captured eye" />
          <p className="hint">Review the image. Is the eye sharp, centered and well lit?</p>
          <div className="btn-row">
            <button className="btn primary" onClick={() => onUse(shot.file)}>✅ Use Image</button>
            <button className="btn" onClick={retake}>↩ Retake</button>
          </div>
        </>
      )}
    </div>
  );
}
