import Disclaimer from "../components/Disclaimer.jsx";

export default function About() {
  return (
    <>
      <section className="card">
        <h2>About</h2>
        <p><strong>AI-Based Vision Screening for Spectacle Requirement Using ResNet-18 Transfer Learning</strong> — a final-year
          BE Information Technology project.</p>
        <p>The app preprocesses an eye image, extracts 512 deep features with ResNet-18 and four handcrafted ocular image
          features, fuses them into a 516-dimensional vector and classifies the image as Normal, Myopia or Hyperopia.</p>
        <h3>Limitations</h3>
        <ul>
          <li>It is an image-screening prototype. It does not measure refractive power and never gives an eyeglass prescription.</li>
          <li>Results depend on focus, lighting, pupil size, reflections, camera distance and the training data.</li>
          <li>The training labels are folder labels that have not been clinically verified.</li>
        </ul>
        <h3>Future scope</h3>
        <p>Infrared photorefraction, specialised autorefractor hardware, clinical validation, a larger real-world dataset,
          smartphone calibration and multi-device validation.</p>
      </section>
      <Disclaimer />
    </>
  );
}
