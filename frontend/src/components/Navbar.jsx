import { Link } from 'react-router-dom'
import { FaBars, FaTimes } from 'react-icons/fa'
import { useState } from 'react'
import biobuddyLogo from '../assets/images/biobuddylogo.png'
import '../App.css'

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false)

  const toggleMenu = () => {
    setIsOpen(!isOpen)
  }

  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-logo">
          <img src={biobuddyLogo} alt="BioBuddy" className="nav-logo-img" />
          <span className="nav-logo-text">BioBuddy</span>
        </Link>
        
        <div className={`nav-menu ${isOpen ? 'active' : ''}`}>
          <Link to="/" className="nav-link" onClick={() => setIsOpen(false)}>
            Home
          </Link>
          <Link to="/main" className="nav-link" onClick={() => setIsOpen(false)}>
            About
          </Link>
          <Link to="/technology" className="nav-link" onClick={() => setIsOpen(false)}>
            Technology
          </Link>
          <Link to="/contact" className="nav-link" onClick={() => setIsOpen(false)}>
            Contact
          </Link>
          <Link to="/login" className="nav-link" onClick={() => setIsOpen(false)}>
            Login
          </Link>
        </div>
        
        <div className="nav-toggle" onClick={toggleMenu}>
          {isOpen ? <FaTimes /> : <FaBars />}
        </div>
      </div>
    </nav>
  )
}

export default Navbar