import '../App.css'
import { FaBrain, FaWifi, FaBatteryFull, FaShieldAlt, FaEye, FaMicrochip, FaCloud, FaMobile } from 'react-icons/fa'

const TechnologyPage = () => {
  return (
    <div className="page">
      <div className="technology-content">
        <h1>Our Technology</h1>
        
        <p style={{ fontSize: '1.2rem', textAlign: 'center', marginBottom: '3rem', color: 'var(--gray-600)' }}>
          BioBuddy combines cutting-edge artificial intelligence with rugged outdoor engineering
          to create the most advanced wildlife detection system available.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', margin: '3rem 0' }}>
          <div style={{ padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px', border: '2px solid var(--forest-green)' }}>
            <FaBrain style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>AI-Powered Detection</h3>
            <p>Custom TensorFlow Lite models trained on over 100,000 wildlife images and videos, achieving 99.7% detection accuracy across 50+ species.</p>
          </div>
          
          <div style={{ padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px', border: '2px solid var(--forest-green)' }}>
            <FaWifi style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>100% Offline Operation</h3>
            <p>All AI processing happens locally on the device. No internet connection required - works in the most remote wilderness locations.</p>
          </div>
          
          <div style={{ padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px', border: '2px solid var(--forest-green)' }}>
            <FaBatteryFull style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Extended Battery Life</h3>
            <p>Optimized power management provides up to 30 days of continuous operation on a single charge, with solar charging capability.</p>
          </div>
          
          <div style={{ padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px', border: '2px solid var(--forest-green)' }}>
            <FaShieldAlt style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Rugged Design</h3>
            <p>IP67 waterproof rating, -40°F to 140°F temperature range, and military-grade durability for any outdoor environment.</p>
          </div>
        </div>

        <h2 style={{ color: 'var(--forest-green)', marginTop: '3rem', marginBottom: '1.5rem' }}>Technical Specifications</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1.5rem', margin: '2rem 0' }}>
          <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px', border: '1px solid var(--gray-200)', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Processor</h4>
            <p>Quad-core ARM Cortex-A72 @ 1.8GHz with dedicated AI accelerator</p>
          </div>
          
          <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px', border: '1px solid var(--gray-200)', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Memory</h4>
            <p>4GB LPDDR4 RAM, 32GB eMMC storage for AI models and data</p>
          </div>
          
          <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px', border: '1px solid var(--gray-200)', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Sensors</h4>
            <p>4K camera, thermal imaging, motion sensors, GPS, environmental sensors</p>
          </div>
          
          <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px', border: '1px solid var(--gray-200)', boxShadow: '0 2px 4px rgba(0,0,0,0.1)' }}>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Connectivity</h4>
            <p>WiFi 6, Bluetooth 5.0, LTE (optional), satellite communication (optional)</p>
          </div>
        </div>

        <h2 style={{ color: 'var(--forest-green)', marginTop: '3rem', marginBottom: '1.5rem' }}>AI Detection Capabilities</h2>
        
        <div style={{ background: 'var(--light-green)', padding: '2rem', borderRadius: '12px', marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Supported Species</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
              <strong>Bears:</strong> Grizzly, Black, Polar, Brown
            </div>
            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
              <strong>Big Cats:</strong> Mountain Lions, Lynx, Bobcats
            </div>
            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
              <strong>Wolves:</strong> Gray Wolf, Coyote, Fox
            </div>
            <div style={{ background: 'white', padding: '1rem', borderRadius: '8px' }}>
              <strong>Other:</strong> Moose, Elk, Deer, Wild Boar
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', margin: '2rem 0' }}>
          <div style={{ padding: '2rem', background: 'white', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
            <FaEye style={{ fontSize: '2.5rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Detection Range</h4>
            <p>Up to 300 feet in daylight, 150 feet at night with thermal imaging</p>
          </div>
          
          <div style={{ padding: '2rem', background: 'white', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
            <FaMicrochip style={{ fontSize: '2.5rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Processing Speed</h4>
            <p>Real-time detection with &lt; 100ms response time for immediate alerts</p>
          </div>
          
          <div style={{ padding: '2rem', background: 'white', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)' }}>
            <FaCloud style={{ fontSize: '2.5rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Data Storage</h4>
            <p>Local storage for 30 days of detection events and environmental data</p>
          </div>
        </div>

        <h2 style={{ color: 'var(--forest-green)', marginTop: '3rem', marginBottom: '1.5rem' }}>How It Works</h2>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem', margin: '2rem 0' }}>
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--forest-green)', marginBottom: '1rem' }}>1</div>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Continuous Monitoring</h4>
            <p>AI-powered cameras and sensors scan the environment 24/7 for wildlife movement</p>
          </div>
          
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--forest-green)', marginBottom: '1rem' }}>2</div>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>AI Analysis</h4>
            <p>Advanced neural networks analyze visual and thermal data to identify wildlife species</p>
          </div>
          
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--forest-green)', marginBottom: '1rem' }}>3</div>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Instant Alerts</h4>
            <p>Immediate notifications via voice, light, and mobile app when threats are detected</p>
          </div>
          
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <div style={{ fontSize: '3rem', fontWeight: 'bold', color: 'var(--forest-green)', marginBottom: '1rem' }}>4</div>
            <h4 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Data Insights</h4>
            <p>Comprehensive logging and analytics for wildlife patterns and safety insights</p>
          </div>
        </div>

        <h2 style={{ color: 'var(--forest-green)', marginTop: '3rem', marginBottom: '1.5rem' }}>Mobile App Integration</h2>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem', background: 'var(--light-green)', padding: '2rem', borderRadius: '12px', marginBottom: '2rem' }}>
          <FaMobile style={{ fontSize: '4rem', color: 'var(--forest-green)' }} />
          <div>
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>BioBuddy Mobile App</h3>
            <ul style={{ listStyle: 'none', padding: 0 }}>
              <li style={{ marginBottom: '0.5rem' }}>• Real-time alerts and notifications</li>
              <li style={{ marginBottom: '0.5rem' }}>• Live camera feed and thermal imaging</li>
              <li style={{ marginBottom: '0.5rem' }}>• Wildlife detection history and analytics</li>
              <li style={{ marginBottom: '0.5rem' }}>• Emergency contact integration</li>
              <li style={{ marginBottom: '0.5rem' }}>• Offline maps and GPS tracking</li>
            </ul>
          </div>
        </div>

        <div style={{ background: 'var(--gray-100)', padding: '2rem', borderRadius: '12px', textAlign: 'center' }}>
          <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Ready to Experience the Future of Outdoor Safety?</h3>
          <p style={{ marginBottom: '1.5rem' }}>
            BioBuddy's advanced AI technology is ready to protect you in any wilderness environment.
            Join thousands of outdoor enthusiasts who trust BioBuddy for their safety.
          </p>
          <button className="btn" style={{ background: 'var(--forest-green)', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '50px', cursor: 'pointer' }}>
            Get BioBuddy Today
          </button>
        </div>
      </div>
    </div>
  )
}

export default TechnologyPage