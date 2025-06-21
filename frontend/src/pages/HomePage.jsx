import '../App.css'
import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { FaArrowRight, FaTree, FaBrain, FaShieldAlt } from 'react-icons/fa'

const HomePage = () => {
  const [isVisible, setIsVisible] = useState(false)
  const [showButton, setShowButton] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    // Animate elements in sequence
    setTimeout(() => setIsVisible(true), 500)
    setTimeout(() => setShowButton(true), 2000)
  }, [])

  const handleEnter = () => {
    // Add a smooth transition effect
    document.body.style.opacity = '0'
    document.body.style.transition = 'opacity 0.5s ease'
    
    setTimeout(() => {
      navigate('/main')
    }, 500)
  }

  return (
    <div className="entry-page">
      <div className="entry-background">
        <div className="forest-overlay"></div>
        <div className="stars"></div>
      </div>
      
      <div className="entry-content">
        <div className={`entry-text ${isVisible ? 'visible' : ''}`}>
          <h1 className="main-title">
            <span className="title-line">When Nature</span>
            <span className="title-line">Meets AI</span>
          </h1>
          
          <p className="subtitle">
            Welcome to the future of wilderness exploration
          </p>
          
          <div className="features-preview">
            <div className="feature-item">
              <FaTree className="feature-icon" />
              <span>Wilderness Protection</span>
            </div>
            <div className="feature-item">
              <FaBrain className="feature-icon" />
              <span>AI-Powered Detection</span>
            </div>
            <div className="feature-item">
              <FaShieldAlt className="feature-icon" />
              <span>24/7 Safety</span>
            </div>
          </div>
        </div>
        
        <div className={`enter-button-container ${showButton ? 'visible' : ''}`}>
          <button className="enter-button" onClick={handleEnter}>
            <span>Enter BioBuddy</span>
            <FaArrowRight className="arrow-icon" />
          </button>
          <p className="enter-subtitle">Your AI Wilderness Companion</p>
        </div>
      </div>
    </div>
  )
}

export default HomePage