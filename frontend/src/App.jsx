import { useState } from "react";
import Navbar from "./components/Navbar.jsx";
import Home from "./pages/Home.jsx";
import Screen from "./pages/Screen.jsx";
import History from "./pages/History.jsx";
import ModelInfo from "./pages/ModelInfo.jsx";
import About from "./pages/About.jsx";
import EyeCare from "./pages/EyeCare.jsx";
import VisionTests from "./pages/VisionTests.jsx";

export default function App() {
  const [page, setPage] = useState("home");
  const [intent, setIntent] = useState(null); // "upload" | "camera" | null

  const go = (p, i = null) => {
    setIntent(i);
    setPage(p);
    window.scrollTo(0, 0);
  };

  return (
    <>
      <Navbar page={page} go={go} />
      <main className="container">
        {page === "home" && <Home go={go} />}
        {page === "screen" && <Screen key={intent || "screen"} intent={intent} />}
        {page === "care" && <EyeCare go={go} />}
        {page === "tests" && <VisionTests go={go} />}
        {page === "history" && <History />}
        {page === "model" && <ModelInfo />}
        {page === "about" && <About />}
      </main>
      <footer className="footer">
        AI-based preliminary screening prototype. Not a medical diagnosis. Consult an eye-care professional.
      </footer>
    </>
  );
}
