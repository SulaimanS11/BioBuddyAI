import '../App.css'
import Button from '../components/Button'
import { FaTree, FaWifi, FaBell, FaBrain, FaCogs, FaShieldAlt, FaCode, FaCheck, FaTimes, FaStar, FaBatteryFull, FaEye } from 'react-icons/fa'
import { useNavigate } from 'react-router-dom'
import { useEffect, useRef } from 'react'

const MainPage = () => {
  const navigate = useNavigate()

  // Scroll-triggered animation logic
  const sectionRefs = useRef([])
  const scrollContainerRef = useRef(null)

  useEffect(() => {
    const observer = new window.IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible')
          }
        })
      },
      { threshold: 0.2 }
    )
    sectionRefs.current.forEach(ref => {
      if (ref) observer.observe(ref)
    })
    return () => {
      sectionRefs.current.forEach(ref => {
        if (ref) observer.unobserve(ref)
      })
    }
  }, [])

  useEffect(() => {
    const handleWheel = (e) => {
      if (scrollContainerRef.current) {
        e.preventDefault()
        scrollContainerRef.current.scrollLeft += e.deltaY
      }
    }

    const container = scrollContainerRef.current
    if (container) {
      container.addEventListener('wheel', handleWheel, { passive: false })
    }

    return () => {
      if (container) {
        container.removeEventListener('wheel', handleWheel)
      }
    }
  }, [])

  const handleGetStarted = () => {
    navigate('/register')
  }

  return (
    <div className="page">
      {/* Hero Section */}
      <section 
        className="hero section"
        ref={el => (sectionRefs.current[0] = el)}
        style={{ backgroundImage: `linear-gradient(rgba(0,0,0,0.4), rgba(0,0,0,0.4)), url('/src/assets/images/forest-bg.jpg')` }}
      >
        <div className="hero-content">
          <h1>Your AI Wilderness Companion</h1>
          <p>Advanced AI-powered wildlife detection for outdoor safety</p>
          <div className="hero-buttons">
            <Button onClick={handleGetStarted}>Get Started</Button>
            <Button variant="outline">See Demo</Button>
          </div>
        </div>
      </section>

      {/* Who Are We Section */}
      <section className="who-are-we section" ref={el => (sectionRefs.current[1] = el)}>
        <h2>Who Are We</h2>
        <div className="who-are-we-content">
          <div className="company-intro">
            <h3>Our Mission</h3>
            <p>We are a team of passionate technologists, outdoor enthusiasts, and safety advocates dedicated to revolutionizing wilderness safety through cutting-edge AI and quantum computing technology.</p>
          </div>
          <div className="team-grid">
            <div className="team-card">
              <FaBrain className="team-icon" />
              <h3>AI & Machine Learning</h3>
              <p>Our AI specialists develop neural networks trained on thousands of wilderness scenarios for accurate wildlife detection and pattern recognition.</p>
            </div>
            <div className="team-card">
              <FaCode className="team-icon" />
              <h3>Quantum Computing</h3>
              <p>Quantum engineers implement real Qiskit circuits for advanced threat assessment using quantum entanglement and superposition.</p>
            </div>
            <div className="team-card">
              <FaTree className="team-icon" />
              <h3>Wilderness Experts</h3>
              <p>Outdoor professionals and wildlife biologists ensure our technology meets real-world wilderness challenges and safety requirements.</p>
            </div>
            <div className="team-card">
              <FaShieldAlt className="team-icon" />
              <h3>Safety & Security</h3>
              <p>Cybersecurity experts and safety engineers guarantee your data protection and reliable operation in remote environments.</p>
            </div>
            <div className="team-card">
              <FaCogs className="team-icon" />
              <h3>Hardware Engineering</h3>
              <p>Our engineers design rugged, weatherproof devices that withstand the harshest outdoor conditions while maintaining peak performance.</p>
            </div>
            <div className="team-card">
              <FaWifi className="team-icon" />
              <h3>Connectivity & IoT</h3>
              <p>IoT specialists ensure seamless communication and offline processing capabilities for remote wilderness locations.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Why Section */}
      <section style={{ 
        background: 'linear-gradient(135deg, var(--forest-green), var(--pine-green))', 
        color: 'white', 
        padding: '4rem 2rem',
        position: 'relative',
        overflow: 'hidden'
      }}>
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.25)',
          zIndex: 1
        }}></div>
        
        <div style={{ position: 'relative', zIndex: 2, maxWidth: '1200px', margin: '0 auto' }}>
          <h2 style={{ textAlign: 'center', marginBottom: '1rem', fontSize: '2.5rem', fontWeight: 'bold' }}>
            Why Did We Create BioBuddy?
          </h2>
          
          <div style={{ 
            background: 'rgba(255, 255, 255, 0.1)', 
            padding: '1.5rem', 
            borderRadius: '12px', 
            marginBottom: '2rem',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.2)'
          }}>
            <p style={{ 
              textAlign: 'center', 
              fontSize: '1.1rem', 
              lineHeight: '1.6',
              margin: 0,
              fontStyle: 'italic'
            }}>
              "My friend James was hiking in the mountains when his companion got bitten by a snake. 
              They had no idea what type of snake it was, how dangerous it could be, or what to do next. 
              The fear and uncertainty in that moment made me realize we needed a better way to identify 
              wildlife threats in real-time. That's when BioBuddy was born."
            </p>
          </div>
          
          <p style={{ textAlign: 'center', marginBottom: '3rem', fontSize: '1.2rem', opacity: 0.9 }}>
            Every year, thousands of outdoor enthusiasts face dangerous wildlife encounters without proper preparation or knowledge.
          </p>
          
          <div className="why-content">
            <div className="problem-statement">
              <h3>The Problem We're Solving</h3>
              <p>Traditional safety measures are often inadequate, leaving hikers, campers, and outdoor workers vulnerable to unexpected wildlife encounters.</p>
            </div>
            
            <div className="motivation-cards">
              <div className="motivation-card">
                <div className="card-icon">🐍</div>
                <h4>Snake Bites</h4>
                <p>Over 7,000 venomous snake bites occur annually in the US alone, with many victims unable to identify the species.</p>
              </div>
              
              <div className="motivation-card">
                <div className="card-icon">🐻</div>
                <h4>Bear Encounters</h4>
                <p>Bear attacks are rare but devastating, often occurring when hikers are unaware of nearby wildlife.</p>
              </div>
              
              <div className="motivation-card">
                <div className="card-icon">🌿</div>
                <h4>Poisonous Plants</h4>
                <p>Many hikers accidentally come into contact with toxic plants, leading to severe reactions.</p>
              </div>
            </div>
            
            <div className="our-vision">
              <h3>Our Vision</h3>
              <p>We envision a world where anyone can safely explore the wilderness without fear of unexpected wildlife encounters. By combining cutting-edge AI with quantum computing, we're creating a new standard for outdoor safety that's both intelligent and accessible.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Why Choose BioBuddy Section */}
      <section className="why-choose-bio-buddy section" ref={el => (sectionRefs.current[3] = el)}>
        <h2>Why Choose BioBuddy?</h2>
        
        <p style={{ textAlign: 'center', marginBottom: '2rem', fontSize: '1.1rem', color: 'var(--gray-600)' }}>
          See how BioBuddy compares to traditional wildlife safety solutions
        </p>

        <div className="comparison-scroll-container" ref={scrollContainerRef}>
          <div className="comparison-slide">
            <div className="slide-header">
              <h3>Basic Motion Sensors</h3>
              <div className="competitor-icon">📡</div>
            </div>
            <div className="comparison-content">
              <div className="pros-cons">
                <div className="pros">
                  <h4>Pros</h4>
                  <ul>
                    <li><FaCheck style={{ color: 'green' }} /> Detects movement</li>
                    <li><FaCheck style={{ color: 'green' }} /> Some early warning</li>
                    <li><FaCheck style={{ color: 'green' }} /> Affordable</li>
                  </ul>
                </div>
                <div className="cons">
                  <h4>Cons</h4>
                  <ul>
                    <li><FaTimes style={{ color: 'red' }} /> False alarms</li>
                    <li><FaTimes style={{ color: 'red' }} /> No species identification</li>
                    <li><FaTimes style={{ color: 'red' }} /> Limited range</li>
                    <li><FaTimes style={{ color: 'red' }} /> No threat level assessment</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          <div className="comparison-slide bio-buddy-highlight">
            <div className="slide-header">
              <h3>BioBuddy AI System</h3>
              <div className="bio-buddy-icon">🤖</div>
            </div>
            <div className="comparison-content">
              <div className="bio-buddy-features">
                <div className="feature">
                  <FaBrain style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>AI-powered species identification</span>
                </div>
                <div className="feature">
                  <FaShieldAlt style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>Real-time threat assessment</span>
                </div>
                <div className="feature">
                  <FaBatteryFull style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>Extended battery life</span>
                </div>
                <div className="feature">
                  <FaWifi style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>100% offline operation</span>
                </div>
                <div className="feature">
                  <FaStar style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>High detection accuracy</span>
                </div>
                <div className="feature">
                  <FaCheck style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>Multiple species supported</span>
                </div>
                <div className="feature">
                  <FaEye style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                  <span>48-hour continuous detection</span>
                </div>
              </div>
              <div className="bio-buddy-advantage">
                <h4>Why BioBuddy Wins:</h4>
                <p>Combines the best of all solutions with advanced AI, quantum computing, and rugged design for the ultimate wilderness safety experience.</p>
              </div>
            </div>
          </div>

          <div className="comparison-slide">
            <div className="slide-header">
              <h3>Basic Trail Cameras</h3>
              <div className="competitor-icon">📷</div>
            </div>
            <div className="comparison-content">
              <div className="pros-cons">
                <div className="pros">
                  <h4>Pros</h4>
                  <ul>
                    <li><FaCheck style={{ color: 'green' }} /> Visual confirmation</li>
                    <li><FaCheck style={{ color: 'green' }} /> Records activity</li>
                    <li><FaCheck style={{ color: 'green' }} /> Good image quality</li>
                  </ul>
                </div>
                <div className="cons">
                  <h4>Cons</h4>
                  <ul>
                    <li><FaTimes style={{ color: 'red' }} /> No real-time alerts</li>
                    <li><FaTimes style={{ color: 'red' }} /> Requires manual review</li>
                    <li><FaTimes style={{ color: 'red' }} /> Limited battery life</li>
                    <li><FaTimes style={{ color: 'red' }} /> No AI analysis</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="testimonials section" ref={el => (sectionRefs.current[4] = el)}>
        <h2>Trusted by Outdoor Enthusiasts</h2>
        <div className="testimonial-cards">
          <div className="testimonial">
            <p>"BioBuddy's detection warned me of a bear approaching - it's like having extra eyes in the wilderness!"</p>
            <div className="author">- Dr. Sarah Chen, Wildlife Expert</div>
          </div>
          <div className="testimonial">
            <p>"As a park ranger, I recommend BioBuddy to all visitors. The AI processing is revolutionary for safety."</p>
            <div className="author">- Mark, National Park Ranger</div>
          </div>
          <div className="testimonial">
            <p>"The quantum threat assessment is incredibly accurate. It's the future of wilderness safety technology."</p>
            <div className="author">- Dr. Alex Rodriguez, Quantum Computing Researcher</div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default MainPage 