import '../App.css'
import { FaUsers, FaAward, FaGlobe, FaHeart } from 'react-icons/fa'

const AboutPage = () => {
  return (
    <div className="page">
      <div className="about-content">
        <h1>Who We Are</h1>
        
        <p>
          Founded by outdoor enthusiasts and AI experts, BioBuddy combines wilderness
          experience with cutting-edge technology to create the most advanced wildlife
          detection system available.
        </p>
        
        <p>
          Our team brings together decades of experience in artificial intelligence,
          wildlife biology, and outdoor safety. We understand the challenges that
          nature lovers face in the wilderness, and we've built BioBuddy to address
          those challenges head-on.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '2rem', margin: '3rem 0' }}>
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <FaUsers style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3>Expert Team</h3>
            <p>AI researchers, wildlife biologists, and outdoor safety experts working together.</p>
          </div>
          
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <FaAward style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3>Proven Track Record</h3>
            <p>Over 10,000 successful deployments in national parks and wilderness areas.</p>
          </div>
          
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <FaGlobe style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3>Global Impact</h3>
            <p>Protecting outdoor enthusiasts in over 25 countries worldwide.</p>
          </div>
          
          <div style={{ textAlign: 'center', padding: '2rem', background: 'var(--gray-100)', borderRadius: '12px' }}>
            <FaHeart style={{ fontSize: '3rem', color: 'var(--forest-green)', marginBottom: '1rem' }} />
            <h3>Passion for Nature</h3>
            <p>We love the outdoors and want to help others enjoy it safely.</p>
          </div>
        </div>

        <h2 style={{ color: 'var(--forest-green)', marginTop: '3rem', marginBottom: '1.5rem' }}>Our Story</h2>
        
        <p>
          BioBuddy was born from a real-life encounter in the backcountry. Our founder,
          Dr. Sarah Chen, was camping alone in a remote area when she had a close
          encounter with a bear. While she was prepared and the situation ended safely,
          it made her realize how many outdoor enthusiasts could benefit from advanced
          warning systems.
        </p>
        
        <p>
          Teaming up with AI researcher Dr. Michael Rodriguez and wildlife biologist
          Dr. Emily Thompson, they began developing what would become BioBuddy. The
          goal was simple: create a device that could detect wildlife threats in
          real-time, work completely offline, and be rugged enough for any outdoor
          environment.
        </p>
        
        <p>
          After three years of development, testing in some of the most challenging
          environments, and partnerships with national parks and outdoor organizations,
          BioBuddy is now helping protect thousands of outdoor enthusiasts around the
          world.
        </p>

        <h2 style={{ color: 'var(--forest-green)', marginTop: '3rem', marginBottom: '1.5rem' }}>Our Values</h2>
        
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
  )
}

export default AboutPage