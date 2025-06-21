import '../App.css'

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'medium',
  icon = null,
  disabled = false,
  onClick,
  className = '',
  ...props 
}) => {
  const variants = {
    primary: {
      background: 'var(--forest-green)',
      color: 'white',
      hoverBackground: 'var(--pine-green)'
    },
    secondary: {
      background: 'white',
      color: 'var(--forest-green)',
      border: '2px solid var(--forest-green)',
      hoverBackground: 'var(--light-green)'
    },
    outline: {
      background: 'transparent',
      color: 'white',
      border: '2px solid white',
      hoverBackground: 'white',
      hoverColor: 'var(--forest-green)'
    },
    danger: {
      background: '#e53e3e',
      color: 'white',
      hoverBackground: '#c53030'
    }
  }

  const sizes = {
    small: {
      padding: '0.5rem 1rem',
      fontSize: '0.875rem'
    },
    medium: {
      padding: '0.75rem 1.5rem',
      fontSize: '1rem'
    },
    large: {
      padding: '1rem 2rem',
      fontSize: '1.125rem'
    }
  }

  const currentVariant = variants[variant] || variants.primary
  const currentSize = sizes[size] || sizes.medium

  return (
    <button
      className={`btn ${className}`}
      onClick={onClick}
      disabled={disabled}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '0.5rem',
        borderRadius: '50px',
        fontWeight: '600',
        cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'all 0.3s ease',
        border: currentVariant.border || 'none',
        backgroundColor: currentVariant.background,
        color: currentVariant.color,
        padding: currentSize.padding,
        fontSize: currentSize.fontSize,
        opacity: disabled ? 0.7 : 1,
        ...props.style
      }}
      onMouseEnter={(e) => {
        if (!disabled) {
          e.currentTarget.style.backgroundColor = currentVariant.hoverBackground;
          if (currentVariant.hoverColor) {
            e.currentTarget.style.color = currentVariant.hoverColor;
          }
          e.currentTarget.style.transform = 'translateY(-2px)';
          e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
        }
      }}
      onMouseLeave={(e) => {
        if (!disabled) {
          e.currentTarget.style.backgroundColor = currentVariant.background;
          e.currentTarget.style.color = currentVariant.color;
          e.currentTarget.style.transform = 'translateY(0)';
          e.currentTarget.style.boxShadow = 'none';
        }
      }}
      {...props}
    >
      {icon && <span className="btn-icon">{icon}</span>}
      {children}
    </button>
  )
}

export default Button