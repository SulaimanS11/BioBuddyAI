import { Routes, Route, useLocation } from 'react-router-dom'
import { useEffect } from 'react'
import Navbar from "/src/components/Navbar.jsx";
import HomePage from './pages/HomePage'
import MainPage from './pages/MainPage'
import AboutPage from './pages/AboutPage'
import MissionPage from './pages/MissionPage'
import TechnologyPage from './pages/TechnologyPage'
import ContactPage from './pages/ContactPage'
import './App.css'

function App() {
  const location = useLocation()
  const isEntryPage = location.pathname === '/'

  useEffect(() => {
    // Reset body opacity after the entry page transition
    if (location.pathname !== '/') {
      document.body.style.opacity = '1'
    }
  }, [location])

  return (
    <div className="app">
      {!isEntryPage && <Navbar />}
      <main>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/main" element={<MainPage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route path="/mission" element={<MissionPage />} />
          <Route path="/technology" element={<TechnologyPage />} />
          <Route path="/contact" element={<ContactPage />} />
        </Routes>
      </main>
    </div>
  )
}

export default App