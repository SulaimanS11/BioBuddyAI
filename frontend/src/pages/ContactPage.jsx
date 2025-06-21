import '../App.css'
import { FaEnvelope, FaPhone, FaMapMarkerAlt, FaClock } from 'react-icons/fa'

const ContactPage = () => {
  return (
    <div className="page">
      <div className="contact-content">
        <h1>Contact Us</h1>
        
        <p style={{ fontSize: '1.2rem', textAlign: 'center', marginBottom: '3rem', color: 'var(--gray-600)' }}>
          Have questions about BioBuddy? We're here to help! Reach out to our team
          for support, sales inquiries, or partnership opportunities.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem', marginBottom: '3rem' }}>
          <div className="contact-info">
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>Get in Touch</h3>
            
            <p style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <FaEnvelope style={{ color: 'var(--forest-green)' }} />
              <span><strong>Email:</strong> info@biobuddy.com</span>
            </p>
            
            <p style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <FaPhone style={{ color: 'var(--forest-green)' }} />
              <span><strong>Phone:</strong> (555) 123-4567</span>
            </p>
            
            <p style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <FaMapMarkerAlt style={{ color: 'var(--forest-green)' }} />
              <span><strong>Address:</strong> 123 Wilderness Trail, Boulder, CO 80301</span>
            </p>
            
            <p style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
              <FaClock style={{ color: 'var(--forest-green)' }} />
              <span><strong>Hours:</strong> Mon-Fri 9AM-6PM MST</span>
            </p>
          </div>

          <div className="contact-form">
            <h3 style={{ color: 'var(--forest-green)', marginBottom: '1.5rem' }}>Send us a Message</h3>
            
            <form>
              <div className="form-group">
                <label htmlFor="name">Full Name *</label>
                <input type="text" id="name" name="name" required />
              </div>
              
              <div className="form-group">
                <label htmlFor="email">Email Address *</label>
                <input type="email" id="email" name="email" required />
              </div>
              
              <div className="form-group">
                <label htmlFor="phone">Phone Number</label>
                <input type="tel" id="phone" name="phone" />
              </div>
              
              <div className="form-group">
                <label htmlFor="subject">Subject *</label>
                <select id="subject" name="subject" required>
                  <option value="">Select a subject</option>
                  <option value="sales">Sales Inquiry</option>
                  <option value="support">Technical Support</option>
                  <option value="partnership">Partnership</option>
                  <option value="general">General Question</option>
                </select>
              </div>
              
              <div className="form-group">
                <label htmlFor="message">Message *</label>
                <textarea id="message" name="message" rows="5" required></textarea>
              </div>
              
              <button type="submit" className="btn" style={{ width: '100%', background: 'var(--forest-green)', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '50px', cursor: 'pointer' }}>
                Send Message
              </button>
            </form>
          </div>
        </div>

        <div style={{ background: 'var(--light-green)', padding: '2rem', borderRadius: '12px', marginBottom: '2rem' }}>
          <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem', textAlign: 'center' }}>Frequently Asked Questions</h3>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.5rem' }}>
            <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px' }}>
              <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>How does BioBuddy work?</h4>
              <p>BioBuddy uses advanced AI to detect wildlife in real-time, providing instant alerts through voice, light, and mobile notifications.</p>
            </div>
            
            <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px' }}>
              <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Does it work without internet?</h4>
              <p>Yes! BioBuddy operates completely offline. All AI processing happens locally on the device.</p>
            </div>
            
            <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px' }}>
              <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>What's the battery life?</h4>
              <p>BioBuddy can operate for up to 30 days on a single charge, with optional solar charging capability.</p>
            </div>
            
            <div style={{ background: 'white', padding: '1.5rem', borderRadius: '8px' }}>
              <h4 style={{ color: 'var(--forest-green)', marginBottom: '0.5rem' }}>Is it weatherproof?</h4>
              <p>Absolutely! BioBuddy has an IP67 waterproof rating and can withstand temperatures from -40°F to 140°F.</p>
            </div>
          </div>
        </div>

        <div style={{ textAlign: 'center', background: 'var(--gray-100)', padding: '2rem', borderRadius: '12px' }}>
          <h3 style={{ color: 'var(--forest-green)', marginBottom: '1rem' }}>Join Our Community</h3>
          <p style={{ marginBottom: '1.5rem' }}>
            Follow us on social media for the latest updates, outdoor safety tips, and wildlife stories.
          </p>
          <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
            <button className="btn" style={{ background: '#1877f2', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '50px', cursor: 'pointer' }}>
              Facebook
            </button>
            <button className="btn" style={{ background: '#1da1f2', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '50px', cursor: 'pointer' }}>
              Twitter
            </button>
            <button className="btn" style={{ background: '#e4405f', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '50px', cursor: 'pointer' }}>
              Instagram
            </button>
            <button className="btn" style={{ background: '#0077b5', color: 'white', border: 'none', padding: '0.75rem 1.5rem', borderRadius: '50px', cursor: 'pointer' }}>
              LinkedIn
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ContactPage