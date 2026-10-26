import { useEffect, useState } from "react";
import { clearHistory, getHistory } from "../services/api.js";

export default function History() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState("");

  const load = () => getHistory().then(setRows).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  const clear = async () => {
    if (window.confirm("Delete all prediction history?")) { await clearHistory(); load(); }
  };

  return (
    <section className="card">
      <div className="row-between">
        <h2>Prediction History</h2>
        {rows?.length > 0 && <button className="btn ghost" onClick={clear}>Clear history</button>}
      </div>
      {error && <p className="error-text">{error}</p>}
      {rows && rows.length === 0 && <p className="muted">No predictions yet.</p>}
      {rows?.length > 0 && (
        <div className="table-wrap">
          <table className="table">
            <thead><tr><th>Date &amp; Time</th><th>Image</th><th>Prediction</th><th>Confidence</th><th>Screening Result</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td>{r.created_at}</td>
                  <td>{r.thumbnail && <img className="thumb" src={r.thumbnail} alt="" />} <span className="small">{r.filename}</span></td>
                  <td><strong>{r.prediction}</strong></td>
                  <td>{r.confidence}%</td>
                  <td>{r.screening_result}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="muted small">Only the time, a small thumbnail, the file name and the result are stored locally. No personal medical information.</p>
    </section>
  );
}
