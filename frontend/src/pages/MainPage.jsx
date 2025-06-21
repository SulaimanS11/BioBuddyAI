import '../App.css'
import Button from '../components/Button'
import { FaTree, FaWifi, FaBell, FaBrain } from 'react-icons/fa'

const MainPage = () => {
  return (
    <div className="page">
      {/* Hero Section */}
      <section 
        className="hero"
        style={{ backgroundImage: `linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('/src/assets/images/forest-bg.jpg')` }}
      >
        <div className="hero-content">
          <h1>Your AI Wilderness Companion</h1>
          <p>Advanced offline wildlife detection for outdoor safety</p>
          <div className="hero-buttons">
            <Button>Get Started</Button>
            <Button variant="outline">See Demo</Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features">
        <h2>Why BioBuddy Stands Out</h2>
        <div className="features-grid">
          <div className="feature-card">
            <FaTree className="feature-icon" />
            <h3>Wilderness Ready</h3>
            <p>Designed to withstand harsh outdoor conditions with rugged, weatherproof casing.</p>
          </div>
          <div className="feature-card">
            <FaWifi className="feature-icon" />
            <h3>100% Offline</h3>
            <p>No internet needed - works in the most remote locations without signal.</p>
          </div>
          <div className="feature-card">
            <FaBell className="feature-icon" />
            <h3>Instant Alerts</h3>
            <p>Real-time notifications when wildlife is detected near your campsite.</p>
          </div>
          <div className="feature-card">
            <FaBrain className="feature-icon" />
            <h3>Smart Detection</h3>
            <p>Custom AI models trained on thousands of wilderness scenarios.</p>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="how-it-works">
        <h2>How BioBuddy Works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h3>Set Up Your Device</h3>
              <p>Mount BioBuddy near your campsite with a clear view of approaching paths.</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h3>Detection Begins</h3>
              <p>Our AI continuously scans for wildlife and human movement.</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h3>Get Alerts</h3>
              <p>Receive instant voice and light notifications when threats are detected.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="testimonials">
        <h2>Trusted by Outdoor Enthusiasts</h2>
        <div className="testimonial-cards">
          <div className="testimonial">
            <p>"BioBuddy warned me of a bear approaching my campsite - it literally saved my trip!"</p>
            <div className="author">- Sarah, Backpacker</div>
          </div>
          <div className="testimonial">
            <p>"As a park ranger, I recommend BioBuddy to all visitors for added safety."</p>
            <div className="author">- Mark, National Park Ranger</div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default MainPage 