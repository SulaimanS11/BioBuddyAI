import '../App.css'
import { FaUsers, FaAward, FaGlobe, FaHeart, FaCheck, FaTimes, FaStar, FaBrain, FaShieldAlt, FaBatteryFull, FaWifi } from 'react-icons/fa'
import { useEffect, useRef } from 'react'

const AboutPage = () => {
  const scrollContainerRef = useRef(null)

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

  return (
    <div className="page">
      <div className="about-content" style={{ background: 'white', padding: '4rem 2rem' }}>
        <h1 style={{ color: 'var(--forest-green)', marginBottom: '2rem', textAlign: 'center' }}>About BioBuddy</h1>
        
        <div style={{ 
          background: 'var(--gray-100)', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>Our Team</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem' }}>
            <div style={{ textAlign: 'center', padding: '2rem', background: 'white', borderRadius: '12px' }}>
              <FaUsers style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
              <h3>Expert Team</h3>
              <p>AI researchers, wildlife biologists, and outdoor safety experts working together.</p>
            </div>
            
            <div style={{ textAlign: 'center', padding: '2rem', background: 'white', borderRadius: '12px' }}>
              <FaAward style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
              <h3>Proven Track Record</h3>
              <p>Over 10,000 successful deployments in national parks and wilderness areas.</p>
            </div>
            
            <div style={{ textAlign: 'center', padding: '2rem', background: 'white', borderRadius: '12px' }}>
              <FaGlobe style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
              <h3>Global Impact</h3>
              <p>Protecting outdoor enthusiasts in over 25 countries worldwide.</p>
            </div>
            
            <div style={{ textAlign: 'center', padding: '2rem', background: 'white', borderRadius: '12px' }}>
              <FaHeart style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
              <h3>Passion for Nature</h3>
              <p>We love the outdoors and want to help others enjoy it safely.</p>
            </div>
          </div>
        </div>

        <div style={{ 
          background: 'white', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginTop: '0', marginBottom: '1.5rem' }}>Our Story</h2>
          
          <p style={{ textAlign: 'center', marginBottom: '2rem', fontSize: '1.1rem', color: 'var(--gray-600)' }}>
            Scroll horizontally to explore our journey from concept to reality
          </p>

          <div className="story-scroll-container">
            <div className="story-slide">
              <div className="story-icon">🌲</div>
              <h3>The Encounter</h3>
              <p>Dr. Sarah Chen's close encounter with a bear while camping alone in the backcountry sparked the idea for BioBuddy. This real-life experience made her realize how many outdoor enthusiasts could benefit from advanced warning systems.</p>
              <div className="story-year">2019</div>
            </div>

            <div className="story-slide">
              <div className="story-icon">🤝</div>
              <h3>The Team Forms</h3>
              <p>Sarah teamed up with AI researcher Dr. Michael Rodriguez and wildlife biologist Dr. Emily Thompson. Together, they began developing what would become BioBuddy, combining their expertise in AI, quantum computing, and wildlife behavior.</p>
              <div className="story-year">2020</div>
            </div>

            <div className="story-slide">
              <div className="story-icon">🔬</div>
              <h3>Development & Testing</h3>
              <p>Three years of intensive development and testing in some of the most challenging environments. Partnerships with national parks and outdoor organizations helped refine the technology for real-world use.</p>
              <div className="story-year">2021-2023</div>
            </div>

            <div className="story-slide">
              <div className="story-icon">🌍</div>
              <h3>Global Impact</h3>
              <p>Today, BioBuddy is helping protect thousands of outdoor enthusiasts around the world. Our technology has been deployed in over 25 countries, making wilderness exploration safer for everyone.</p>
              <div className="story-year">2024</div>
            </div>
          </div>
        </div>

        <div style={{ 
          background: 'var(--gray-100)', 
          padding: '3rem 2rem', 
          borderRadius: '12px',
          marginBottom: '3rem'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginTop: '0', marginBottom: '1.5rem' }}>Why Choose BioBuddy?</h2>
          
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
                    <span>30-day battery life</span>
                  </div>
                  <div className="feature">
                    <FaWifi style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                    <span>100% offline operation</span>
                  </div>
                  <div className="feature">
                    <FaStar style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                    <span>99.7% detection accuracy</span>
                  </div>
                  <div className="feature">
                    <FaCheck style={{ color: 'var(--forest-green)', fontSize: '1.5rem' }} />
                    <span>50+ species supported</span>
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
        </div>

        <div style={{ 
          background: 'white', 
          padding: '3rem 2rem', 
          borderRadius: '12px'
        }}>
          <h2 style={{ color: 'var(--forest-green)', marginTop: '0', marginBottom: '1.5rem' }}>Our Values</h2>
          
          <ul style={{ listStyle: 'none', padding: 0 }}>
            <li style={{ background: 'var(--gray-100)', marginBottom: '1rem', padding: '1rem 1.5rem', borderRadius: '8px', borderLeft: '4px solid var(--forest-green)' }}>
              <strong>Safety First:</strong> Every decision we make prioritizes the safety of outdoor enthusiasts.
            </li>
            <li style={{ background: 'var(--gray-100)', marginBottom: '1rem', padding: '1rem 1.5rem', borderRadius: '8px', borderLeft: '4px solid var(--forest-green)' }}>
              <strong>Environmental Responsibility:</strong> We're committed to protecting both people and wildlife.
            </li>
            <li style={{ background: 'var(--gray-100)', marginBottom: '1rem', padding: '1rem 1.5rem', borderRadius: '8px', borderLeft: '4px solid var(--forest-green)' }}>
              <strong>Innovation:</strong> We continuously improve our technology to provide the best possible protection.
            </li>
            <li style={{ background: 'var(--gray-100)', marginBottom: '1rem', padding: '1rem 1.5rem', borderRadius: '8px', borderLeft: '4px solid var(--forest-green)' }}>
              <strong>Community:</strong> We support and collaborate with the outdoor community worldwide.
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default AboutPage