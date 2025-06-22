import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { 
  FaUser, 
  FaSignOutAlt, 
  FaWifi, 
  FaShoppingCart, 
  FaHistory, 
  FaBell,
  FaCog,
  FaPlus,
  FaTrash,
  FaEye,
  FaCamera
} from 'react-icons/fa'
import biobuddyLogo from '../assets/images/biobuddylogo.png'
import '../App.css'

const DashboardPage = () => {
  const [user, setUser] = useState(null)
  const [activeTab, setActiveTab] = useState('devices')
  const [devices, setDevices] = useState([])
  const [showAddDevice, setShowAddDevice] = useState(false)
  const [newDevice, setNewDevice] = useState({ name: '', location: '' })
  const navigate = useNavigate()

  // Scroll-triggered animation logic
  const sectionRefs = useRef([])
  useEffect(() => {
    const observer = new window.IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible')
          }
        })
      },
      { threshold: 0.2 }
    )
    sectionRefs.current.forEach(ref => {
      if (ref) observer.observe(ref)
    })
    return () => {
      sectionRefs.current.forEach(ref => {
        if (ref) observer.unobserve(ref)
      })
    }
  }, [])

  useEffect(() => {
    // Check if user is logged in
    const isLoggedIn = localStorage.getItem('isLoggedIn')
    if (!isLoggedIn) {
      navigate('/login')
      return
    }

    // Get user data
    const userData = JSON.parse(localStorage.getItem('user') || '{}')
    setUser(userData)
  }, [navigate])

  const handleLogout = () => {
    localStorage.removeItem('isLoggedIn')
    navigate('/main')
  }

  const handleAddDevice = () => {
    if (newDevice.name && newDevice.location) {
      const device = {
        id: Date.now(),
        name: newDevice.name,
        status: 'online',
        location: newDevice.location,
        lastSeen: 'Just now'
      }
      setDevices([...devices, device])
      setNewDevice({ name: '', location: '' })
      setShowAddDevice(false)
    }
  }

  const handleRemoveDevice = (deviceId) => {
    setDevices(devices.filter(device => device.id !== deviceId))
  }

  // Updated products with focus on AI and smart detection
  const products = [
    { 
      id: 1, 
      name: 'BioBuddy', 
      price: '$59.99', 
      description: 'Advanced AI wildlife detection system with smart decision engine', 
      inStock: true
    },
    { 
      id: 2, 
      name: 'BioBuddy Mount', 
      price: '$29.99', 
      description: 'Rugged mount with motorized range maintenance and sensors', 
      inStock: true
    },
    { 
      id: 3, 
      name: 'Camera for BioBuddy', 
      price: '$19.99', 
      description: 'Enhanced camera with advanced detection capabilities', 
      inStock: true
    }
  ]

  const detectionHistory = [
    { id: 1, type: 'Bear', location: 'Campsite A', time: '2024-01-15 14:30', severity: 'high', quantum: false },
    { id: 2, type: 'Deer', location: 'Campsite A', time: '2024-01-15 12:15', severity: 'low', quantum: false },
    { id: 3, type: 'Human', location: 'Campsite B', time: '2024-01-14 18:45', severity: 'medium', quantum: false },
    { id: 4, type: 'Wolf', location: 'Campsite C', time: '2024-01-14 16:20', severity: 'high', quantum: false }
  ]

  const renderDevices = () => (
    <div className="dashboard-section" ref={el => (sectionRefs.current[0] = el)}>
      <div className="section-header">
        <h3>Your Devices</h3>
        <button className="add-button" onClick={() => setShowAddDevice(true)}>
          <FaPlus /> Add Device
        </button>
      </div>

      {showAddDevice && (
        <div className="add-device-modal">
          <div className="modal-content">
            <h4>Add New Device</h4>
            <div className="form-group">
              <input
                type="text"
                placeholder="Device Name (e.g., BioBuddy Pro)"
                value={newDevice.name}
                onChange={(e) => setNewDevice({...newDevice, name: e.target.value})}
              />
            </div>
            <div className="form-group">
              <input
                type="text"
                placeholder="Location (e.g., Campsite A)"
                value={newDevice.location}
                onChange={(e) => setNewDevice({...newDevice, location: e.target.value})}
              />
            </div>
            <div className="modal-actions">
              <button className="auth-button" onClick={handleAddDevice}>
                Add Device
              </button>
              <button className="cancel-button" onClick={() => setShowAddDevice(false)}>
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {devices.length === 0 ? (
        <div className="empty-state">
          <FaWifi className="empty-icon" />
          <h4>No Devices Added</h4>
          <p>Add your first BioBuddy device to start monitoring your wilderness area.</p>
          <button className="add-button" onClick={() => setShowAddDevice(true)}>
            <FaPlus /> Add Your First Device
          </button>
        </div>
      ) : (
        <div className="devices-grid">
          {devices.map(device => (
            <div key={device.id} className="device-card">
              <div className="device-header">
                <FaWifi className={`device-icon ${device.status}`} />
                <div className="device-info">
                  <h4>{device.name}</h4>
                  <span className={`status ${device.status}`}>
                    {device.status === 'online' ? 'Online' : 'Offline'}
                  </span>
                </div>
                <div className="device-actions">
                  <button className="action-button">
                    <FaEye />
                  </button>
                  <button className="action-button">
                    <FaCog />
                  </button>
                  <button 
                    className="action-button danger"
                    onClick={() => handleRemoveDevice(device.id)}
                  >
                    <FaTrash />
                  </button>
                </div>
              </div>
              <div className="device-details">
                <p><strong>Location:</strong> {device.location}</p>
                <p><strong>Last Seen:</strong> {device.lastSeen}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )

  const renderProducts = () => (
    <div className="dashboard-section" ref={el => (sectionRefs.current[1] = el)}>
      <div className="section-header">
        <h3>Products for Sale</h3>
      </div>
      <div className="products-grid">
        {products.map(product => (
          <div key={product.id} className="product-card">
            <div className="product-image">
              {product.name.includes('Camera') ? (
                <FaCamera className="product-icon" />
              ) : product.name.includes('Mount') ? (
                <FaCog className="product-icon" />
              ) : (
                <FaWifi className="product-icon" />
              )}
            </div>
            <div className="product-info">
              <h4>{product.name}</h4>
              <p>{product.description}</p>
              <div className="product-meta">
                <span className="price">{product.price}</span>
                <span className={`stock ${product.inStock ? 'in-stock' : 'out-of-stock'}`}>{product.inStock ? '\nIn Stock' : 'Out of Stock'}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )

  const renderHistory = () => (
    <div className="dashboard-section" ref={el => (sectionRefs.current[2] = el)}>
      <div className="section-header">
        <h3>Detection History</h3>
      </div>
      <div className="history-grid">
        {detectionHistory.map(event => (
          <div key={event.id} className={`history-card`}>
            <div className="history-header">
              <FaHistory className="history-icon" />
              <span className={`severity ${event.severity}`}>{event.severity.charAt(0).toUpperCase() + event.severity.slice(1)}</span>
            </div>
            <div className="history-details">
              <p><strong>Type:</strong> {event.type}</p>
              <p><strong>Location:</strong> {event.location}</p>
              <p><strong>Time:</strong> {event.time}</p>
            </div>
          </div>
        ))}
      </div>
      <div className="dashboard-note" style={{marginTop: '2rem', color: '#b0c4de', fontSize: '0.95rem', textAlign: 'center'}}>
        <strong>Note:</strong> Danger level is determined by our backend decision tree, which uses quantum logic for enhanced accuracy.
      </div>
    </div>
  )

  if (!user) {
    return <div className="loading">Loading...</div>
  }

  return (
    <div className="dashboard-container">
      <div className="dashboard-header">
        <div className="dashboard-logo">
          <img src={biobuddyLogo} alt="BioBuddy" className="dashboard-logo-img" />
          <span className="dashboard-logo-text">BioBuddy Dashboard</span>
        </div>
        <div className="user-info">
          <FaUser className="user-icon" />
          <div>
            <h2>Welcome back, {user.firstName}!</h2>
            <p>{user.email}</p>
          </div>
        </div>
        <button className="logout-button" onClick={handleLogout}>
          <FaSignOutAlt /> Logout
        </button>
      </div>

      <div className="dashboard-tabs">
        <button 
          className={`tab-button ${activeTab === 'devices' ? 'active' : ''}`}
          onClick={() => setActiveTab('devices')}
        >
          <FaWifi /> Devices
        </button>
        <button 
          className={`tab-button ${activeTab === 'products' ? 'active' : ''}`}
          onClick={() => setActiveTab('products')}
        >
          <FaShoppingCart /> Products
        </button>
        <button 
          className={`tab-button ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <FaHistory /> Detection History
        </button>
      </div>

      <div className="dashboard-content">
        {activeTab === 'devices' && renderDevices()}
        {activeTab === 'products' && renderProducts()}
        {activeTab === 'history' && renderHistory()}
      </div>
    </div>
  )
}

export default DashboardPage 