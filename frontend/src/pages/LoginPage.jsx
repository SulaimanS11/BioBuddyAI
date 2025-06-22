import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { FaEnvelope, FaLock, FaEye, FaEyeSlash } from 'react-icons/fa'
import '../App.css'

const LoginPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [message, setMessage] = useState('')
  const navigate = useNavigate()

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setMessage('')

    try {
      // Simulate API call for login
      await new Promise(resolve => setTimeout(resolve, 1500))
      
      // Check if user exists in localStorage (simulating backend check)
      const storedUser = JSON.parse(localStorage.getItem('user') || '{}')
      
      if (storedUser.email === formData.email && storedUser.isVerified) {
        // Store login state
        localStorage.setItem('isLoggedIn', 'true')
        setMessage('Login successful! Redirecting...')
        
        setTimeout(() => {
          navigate('/dashboard')
        }, 1500)
      } else {
        setMessage('Invalid credentials or email not verified')
      }
      
    } catch (error) {
      setMessage('Login failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h1>Welcome Back</h1>
          <p>Sign in to your BioBuddy account</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <div className="input-wrapper">
              <FaEnvelope className="input-icon" />
              <input
                type="email"
                name="email"
                placeholder="Email Address"
                value={formData.email}
                onChange={handleInputChange}
                required
              />
            </div>
          </div>

          <div className="form-group">
            <div className="input-wrapper">
              <FaLock className="input-icon" />
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                placeholder="Password"
                value={formData.password}
                onChange={handleInputChange}
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
              >
                {showPassword ? <FaEyeSlash /> : <FaEye />}
              </button>
            </div>
          </div>

          {message && (
            <div className={`message ${message.includes('successful') ? 'success' : 'error'}`}>
              {message}
            </div>
          )}

          <button 
            type="submit" 
            className="auth-button"
            disabled={isLoading}
          >
            {isLoading ? 'Signing In...' : 'Sign In'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            Don't have an account? <Link to="/register">Create Account</Link>
          </p>
        </div>
      </div>
    </div>
  )
}

export default LoginPage 