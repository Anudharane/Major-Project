const ITEMS = [
  ["home", "🏠 Home"],
  ["screen", "👁 Screen Eye"],
  ["care", "🛡️ Eye Care"],
  ["tests", "🧪 Vision Tests"],
  ["history", "📋 History"],
  ["about", "About"],
];

export default function Navbar({ page, go }) {
  return (
    <header className="navbar">
      <div className="brand" onClick={() => go("home")}>
        <span>👁️</span> AI Vision Screening
      </div>
      <nav>
        {ITEMS.map(([id, label]) => (
          <button
            key={id}
            className={page === id ? "nav-btn active" : "nav-btn"}
            onClick={() => go(id)}
          >
            {label}
          </button>
        ))}
      </nav>
    </header>
  );
}
