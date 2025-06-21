import '../App.css'

const Card = ({ 
  children, 
  className = '', 
  padding = '1.5rem',
  shadow = true,
  hover = false,
  ...props 
}) => {
  return (
    <div
      className={`card ${className}`}
      style={{
        background: 'white',
        borderRadius: '12px',
        padding: padding,
        boxShadow: shadow ? '0 4px 6px rgba(0, 0, 0, 0.1)' : 'none',
        transition: hover ? 'all 0.3s ease' : 'none',
        border: '1px solid #e5e7eb',
        ...props.style
      }}
      onMouseEnter={(e) => {
        if (hover) {
          e.currentTarget.style.transform = 'translateY(-4px)';
          e.currentTarget.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.15)';
        }
      }}
      onMouseLeave={(e) => {
        if (hover) {
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = shadow ? '0 4px 6px rgba(0, 0, 0, 0.1)' : 'none';
        }
      }}
      {...props}
    >
      {children}
    </div>
  )
}

export default Card
