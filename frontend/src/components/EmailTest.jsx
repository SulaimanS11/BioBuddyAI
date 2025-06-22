import { useState } from 'react'
import { sendVerificationEmail } from '../services/emailService'

const EmailTest = () => {
  const [email, setEmail] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [message, setMessage] = useState('')
  const [debugInfo, setDebugInfo] = useState('')

  const handleTestEmail = async (e) => {
    e.preventDefault()
    setIsLoading(true)
    setMessage('')
    setDebugInfo('')

    try {
      const token = 'test-token-' + Date.now()
      console.log('Sending test email to:', email)
      console.log('Test token:', token)
      
      const result = await sendVerificationEmail(email, token)
      
      console.log('Email result:', result)
      
      if (result.success) {
        setMessage('✅ Test email sent successfully! Check your inbox.')
        setDebugInfo(`Email sent to: ${email}\nToken: ${token}`)
      } else {
        setMessage('❌ Failed to send email: ' + result.message)
        setDebugInfo('Check browser console for more details')
      }
    } catch (error) {
      console.error('Email test error:', error)
      setMessage('❌ Error: ' + error.message)
      setDebugInfo('Check browser console for error details')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div style={{ padding: '20px', maxWidth: '500px', margin: '0 auto' }}>
      <h3>EmailJS Test</h3>
      <p style={{ color: '#666', marginBottom: '20px' }}>
        Test your EmailJS configuration and template
      </p>
      
      <form onSubmit={handleTestEmail}>
        <input
          type="email"
          placeholder="Enter your email to test"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{ 
            width: '100%', 
            padding: '12px', 
            marginBottom: '15px',
            border: '1px solid #ddd',
            borderRadius: '5px',
            fontSize: '16px'
          }}
          required
        />
        <button 
          type="submit" 
          disabled={isLoading}
          style={{ 
            width: '100%', 
            padding: '12px', 
            background: '#1a5f1a', 
            color: 'white', 
            border: 'none',
            borderRadius: '5px',
            cursor: isLoading ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            fontWeight: '600'
          }}
        >
          {isLoading ? 'Sending...' : 'Send Test Email'}
        </button>
      </form>
      
      {message && (
        <div style={{ 
          marginTop: '20px', 
          padding: '15px', 
          borderRadius: '5px',
          background: message.includes('✅') ? '#d4edda' : '#f8d7da',
          color: message.includes('✅') ? '#155724' : '#721c24',
          border: '1px solid ' + (message.includes('✅') ? '#c3e6cb' : '#f5c6cb')
        }}>
          <strong>{message}</strong>
        </div>
      )}
      
      {debugInfo && (
        <div style={{ 
          marginTop: '15px', 
          padding: '15px', 
          borderRadius: '5px',
          background: '#f8f9fa',
          border: '1px solid #dee2e6',
          fontFamily: 'monospace',
          fontSize: '14px',
          whiteSpace: 'pre-line'
        }}>
          <strong>Debug Info:</strong>
          {debugInfo}
        </div>
      )}
      
      <div style={{ 
        marginTop: '30px', 
        padding: '15px', 
        background: '#e7f3ff', 
        borderRadius: '5px',
        border: '1px solid #b3d9ff'
      }}>
        <h4>EmailJS Configuration:</h4>
        <p><strong>Service ID:</strong> service_1gvbaox</p>
        <p><strong>Template ID:</strong> template_qe9dfh1</p>
        <p><strong>Public Key:</strong> apG8bme_eqbTPE5PO</p>
      </div>
    </div>
  )
}

export default EmailTest 