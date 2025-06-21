import { Link } from 'react-router-dom'
import '../App.css'

const Navbar = () => {
  return (
    <nav className="navbar">
      <Link to="/main" className="logo">BioBuddy</Link>
      <div className="nav-links">
        <Link to="/main">Home</Link>
        <Link to="/about">Who We Are</Link>
        <Link to="/mission">Our Mission</Link>
        <Link to="/technology">Technology</Link>
        <Link to="/contact">Contact</Link>
      </div>
    </nav>
  )
}

export default Navbar