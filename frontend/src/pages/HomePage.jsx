import '../App.css'
import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { FaArrowRight, FaTree, FaShieldAlt, FaCogs, FaAtom } from 'react-icons/fa'
import biobuddyLogo from '../assets/images/biobuddylogo.png'

const HomePage = () => {
  const [isVisible, setIsVisible] = useState(false)
  const [showButton, setShowButton] = useState(false)
  const [scrollY, setScrollY] = useState(0)
  const [isScrolling, setIsScrolling] = useState(false)
  const navigate = useNavigate()
  const pageRef = useRef(null)

  useEffect(() => {
    // Animate elements in sequence with staggered timing
    const timer1 = setTimeout(() => setIsVisible(true), 300)
    const timer2 = setTimeout(() => setShowButton(true), 2500)
    
    // Parallax scroll effect
    const handleScroll = () => {
      if (pageRef.current) {
        const scrolled = window.pageYOffset
        setScrollY(scrolled)
        setIsScrolling(scrolled > 50)
      }
    }

    window.addEventListener('scroll', handleScroll)
    
    return () => {
      clearTimeout(timer1)
      clearTimeout(timer2)
      window.removeEventListener('scroll', handleScroll)
    }
  }, [])

  const handleEnter = () => {
    // Enhanced transition with multiple effects
    const page = pageRef.current
    if (page) {
      page.style.transform = 'scale(0.95) translateY(20px)'
      page.style.opacity = '0'
      page.style.filter = 'blur(10px)'
    }
    
    document.body.style.opacity = '0'
    document.body.style.transition = 'all 0.8s cubic-bezier(0.4, 0, 0.2, 1)'
    
    setTimeout(() => {
      navigate('/main')
    }, 800)
  }

  return (
    <div className="entry-page" ref={pageRef}>
      <div className="entry-background">
        <div 
          className="forest-overlay"
          style={{
            transform: `translateY(${scrollY * 0.3}px)`
          }}
        ></div>
        <div 
          className="stars"
          style={{
            transform: `translateY(${scrollY * 0.1}px)`
          }}
        ></div>
        <div 
          className="quantum-particles"
          style={{
            transform: `translateY(${scrollY * 0.2}px) rotate(${scrollY * 0.05}deg)`
          }}
        ></div>
        <div className="floating-elements">
          <div 
            className="floating-element element-1"
            style={{
              transform: `translateY(${scrollY * 0.4}px) rotate(${scrollY * 0.02}deg)`
            }}
          ></div>
          <div 
            className="floating-element element-2"
            style={{
              transform: `translateY(${scrollY * 0.6}px) rotate(${-scrollY * 0.03}deg)`
            }}
          ></div>
          <div 
            className="floating-element element-3"
            style={{
              transform: `translateY(${scrollY * 0.5}px) rotate(${scrollY * 0.01}deg)`
            }}
          ></div>
        </div>
      </div>
      
      <div className="entry-content">
        <div className={`entry-text ${isVisible ? 'visible' : ''}`}>
          <div className="logo-container animate-fade-in">
            <img src={biobuddyLogo} alt="BioBuddy Logo" className="biobuddy-logo" />
          </div>
          
          <h1 className="main-title">
            <span className="title-line animate-slide-up">Wilderness Safety</span>
            <span className="title-line animate-slide-up delay-1">Meets Quantum AI</span>
          </h1>
          
          <p className={`subtitle animate-fade-in delay-2 ${isVisible ? 'visible' : ''}`}>
            Advanced AI-powered wildlife detection with quantum-enhanced decision making
          </p>
          
          <div className={`features-preview animate-slide-up delay-3 ${isVisible ? 'visible' : ''}`}>
            <div className="feature-item animate-bounce-in" style={{ animationDelay: '0.1s' }}>
              <FaTree className="feature-icon" />
              <span>Wilderness Protection</span>
            </div>
            <div className="feature-item animate-bounce-in" style={{ animationDelay: '0.2s' }}>
              <FaShieldAlt className="feature-icon" />
              <span>Real-Time AI Detection</span>
            </div>
            <div className="feature-item animate-bounce-in" style={{ animationDelay: '0.3s' }}>
              <FaCogs className="feature-icon" />
              <span>Smart Decision Engine</span>
            </div>
            <div className="feature-item animate-bounce-in" style={{ animationDelay: '0.4s' }}>
              <FaAtom className="feature-icon" />
              <span>Quantum Logic Backend</span>
            </div>
          </div>
        </div>
        
        <div className={`enter-button-container animate-slide-up delay-4 ${showButton ? 'visible' : ''}`}>
          <button 
            className={`enter-button ${isScrolling ? 'scrolling' : ''}`} 
            onClick={handleEnter}
          >
            <span>Enter BioBuddy</span>
            <FaArrowRight className="arrow-icon" />
          </button>
          <p className="enter-subtitle animate-fade-in delay-5">Your AI Wilderness Companion</p>
        </div>
      </div>

      {/* Scroll indicator */}
      <div className={`scroll-indicator ${isScrolling ? 'hidden' : ''}`}>
        <div className="scroll-arrow"></div>
        <span>Scroll to explore</span>
      </div>
    </div>
  )
}

export default HomePage