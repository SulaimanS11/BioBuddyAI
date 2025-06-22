import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { FaEnvelope, FaLock, FaUser, FaEye, FaEyeSlash } from 'react-icons/fa'
import { sendVerificationEmail, sendMockVerificationEmail } from '../services/emailService'
import '../App.css'

const RegisterPage = () => {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    firstName: '',
    lastName: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
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
    
    if (formData.password !== formData.confirmPassword) {
      setMessage('Passwords do not match')
      return
    }

    if (formData.password.length < 6) {
      setMessage('Password must be at least 6 characters long')
      return
    }

    setIsLoading(true)
    setMessage('')

    try {
      // Generate verification token
      const verificationToken = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15)
      
      console.log('Generated verification token:', verificationToken)
      
      // Try to send real verification email first
      let emailResult = await sendVerificationEmail(formData.email, verificationToken)
      
      // If real email fails, fall back to mock email
      if (!emailResult.success) {
        console.log('Real email failed, using mock email...')
        emailResult = await sendMockVerificationEmail(formData.email)
      }
      
      if (emailResult.success) {
        setMessage('Registration successful! Please check your email for verification.')
        
        // Store user data in localStorage with verification token
        const userData = {
          email: formData.email,
          firstName: formData.firstName,
          lastName: formData.lastName,
          isVerified: false,
          verificationToken: emailResult.token || verificationToken
        }
        
        localStorage.setItem('user', JSON.stringify(userData))
        
        console.log('User data stored:', userData)
        console.log('Verification link:', `${window.location.origin}/verify-email?token=${emailResult.token || verificationToken}`)
        
        // Redirect to verification page after 3 seconds
        setTimeout(() => {
          navigate('/verify-email')
        }, 3000)
      } else {
        setMessage('Registration failed. Please try again.')
      }
      
    } catch (error) {
      console.error('Registration error:', error)
      setMessage('Registration failed. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <h1>Create Your Account</h1>
          <p>Join BioBuddy and start your wilderness safety journey</p>
        </div>

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <div className="input-wrapper">
              <FaUser className="input-icon" />
              <input
                type="text"
                name="firstName"
                placeholder="First Name"
                value={formData.firstName}
                onChange={handleInputChange}
                required
              />
            </div>
          </div>

          <div className="form-group">
            <div className="input-wrapper">
              <FaUser className="input-icon" />
              <input
                type="text"
                name="lastName"
                placeholder="Last Name"
                value={formData.lastName}
                onChange={handleInputChange}
                required
              />
            </div>
          </div>

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

          <div className="form-group">
            <div className="input-wrapper">
              <FaLock className="input-icon" />
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                placeholder="Confirm Password"
                value={formData.confirmPassword}
                onChange={handleInputChange}
                required
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
              >
                {showConfirmPassword ? <FaEyeSlash /> : <FaEye />}
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
            {isLoading ? 'Creating Account...' : 'Create Account'}
          </button>
        </form>

        <div className="auth-footer">
          <p>
            Already have an account? <Link to="/login">Sign In</Link>
          </p>
        </div>
      </div>
    </div>
  )
}

export default RegisterPage 