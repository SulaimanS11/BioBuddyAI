import { useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { FaEnvelope, FaCheckCircle, FaSpinner, FaExclamationTriangle } from 'react-icons/fa'
import '../App.css'

const VerifyEmailPage = () => {
  const [isVerifying, setIsVerifying] = useState(false)
  const [isVerified, setIsVerified] = useState(false)
  const [userEmail, setUserEmail] = useState('')
  const [verificationStatus, setVerificationStatus] = useState('pending') // pending, success, error
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()

  useEffect(() => {
    // Get user email from localStorage
    const user = JSON.parse(localStorage.getItem('user') || '{}')
    setUserEmail(user.email || 'user@example.com')
    
    // Check if there's a verification token in the URL
    const token = searchParams.get('token')
    if (token) {
      handleTokenVerification(token)
    }
  }, [searchParams])

  const handleTokenVerification = async (token) => {
    setIsVerifying(true)
    setVerificationStatus('pending')
    
    try {
      // Get user data from localStorage
      const user = JSON.parse(localStorage.getItem('user') || '{}')
      
      console.log('Verifying token:', token)
      console.log('Stored user:', user)
      console.log('Stored verification token:', user.verificationToken)
      
      // For now, accept any token that looks valid (in a real app, you'd verify against backend)
      if (token && token.length > 10) {
        // Update user verification status
        user.isVerified = true
        localStorage.setItem('user', JSON.stringify(user))
        
        setVerificationStatus('success')
        setIsVerified(true)
        
        console.log('Token verification successful!')
        
        // Redirect to dashboard after 2 seconds
        setTimeout(() => {
          navigate('/dashboard')
        }, 2000)
      } else {
        console.log('Token validation failed - token too short or invalid')
        setVerificationStatus('error')
      }
      
    } catch (error) {
      console.error('Verification failed:', error)
      setVerificationStatus('error')
    } finally {
      setIsVerifying(false)
    }
  }

  const handleVerifyEmail = async () => {
    setIsVerifying(true)
    
    try {
      // Simulate email verification process
      await new Promise(resolve => setTimeout(resolve, 2000))
      
      // Update user verification status
      const user = JSON.parse(localStorage.getItem('user') || '{}')
      user.isVerified = true
      localStorage.setItem('user', JSON.stringify(user))
      
      setIsVerified(true)
      setVerificationStatus('success')
      
      // Redirect to dashboard after 2 seconds
      setTimeout(() => {
        navigate('/dashboard')
      }, 2000)
      
    } catch (error) {
      console.error('Verification failed:', error)
      setVerificationStatus('error')
    } finally {
      setIsVerifying(false)
    }
  }

  const handleResendEmail = async () => {
    setIsVerifying(true)
    
    try {
      // Simulate resending verification email
      await new Promise(resolve => setTimeout(resolve, 1000))
      alert('Verification email resent!')
    } catch (error) {
      alert('Failed to resend email. Please try again.')
    } finally {
      setIsVerifying(false)
    }
  }

  const renderVerificationContent = () => {
    if (verificationStatus === 'success') {
      return (
        <div className="verification-success">
          <FaCheckCircle className="success-icon" />
          <h3>Email Verified!</h3>
          <p>Your account has been successfully verified. Redirecting to dashboard...</p>
        </div>
      )
    }

    if (verificationStatus === 'error') {
      return (
        <div className="verification-error">
          <FaExclamationTriangle className="error-icon" />
          <h3>Verification Failed</h3>
          <p>The verification link is invalid or has expired. Please try again.</p>
          <button className="auth-button" onClick={handleResendEmail}>
            Resend Verification Email
          </button>
        </div>
      )
    }

    return (
      <>
        <div className="verification-steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-text">Check your email inbox</div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-text">Click the verification link</div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-text">Start using BioBuddy</div>
          </div>
        </div>

        <div className="verification-actions">
          <button 
            className="auth-button"
            onClick={handleVerifyEmail}
            disabled={isVerifying}
          >
            {isVerifying ? (
              <>
                <FaSpinner className="spinner" />
                Verifying...
              </>
            ) : (
              "I've Verified My Email"
            )}
          </button>
          
          <button 
            className="resend-button"
            onClick={handleResendEmail}
            disabled={isVerifying}
          >
            Resend Verification Email
          </button>
        </div>
      </>
    )
  }

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-header">
          <div className="verification-icon">
            <FaEnvelope />
          </div>
          <h1>Verify Your Email</h1>
          <p>We've sent a verification link to:</p>
          <p className="user-email">{userEmail}</p>
        </div>

        <div className="verification-content">
          {renderVerificationContent()}
        </div>

        <div className="auth-footer">
          <p>
            Didn't receive the email? Check your spam folder or{' '}
            <button 
              className="link-button"
              onClick={handleResendEmail}
              disabled={isVerifying}
            >
              resend
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

export default VerifyEmailPage 