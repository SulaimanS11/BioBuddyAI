import emailjs from '@emailjs/browser';

// EmailJS configuration
const EMAILJS_CONFIG = {
  SERVICE_ID: 'service_1gvbaox', // Your EmailJS service ID
  TEMPLATE_ID: 'template_qe9dfh1', // Your EmailJS template ID
  PUBLIC_KEY: 'apG8bme_eqbTPE5PO', // Your EmailJS public key
};

// Initialize EmailJS
emailjs.init(EMAILJS_CONFIG.PUBLIC_KEY);

export const sendVerificationEmail = async (userEmail, verificationToken) => {
  try {
    const templateParams = {
      to_email: userEmail,
      verification_link: `${window.location.origin}/verify-email?token=${verificationToken}`,
      user_email: userEmail,
    };

    const response = await emailjs.send(
      EMAILJS_CONFIG.SERVICE_ID,
      EMAILJS_CONFIG.TEMPLATE_ID,
      templateParams
    );

    console.log('Verification email sent successfully:', response);
    return { success: true, message: 'Verification email sent!' };
  } catch (error) {
    console.error('Failed to send verification email:', error);
    return { success: false, message: 'Failed to send verification email' };
  }
};

export const sendWelcomeEmail = async (userEmail, firstName) => {
  try {
    const templateParams = {
      to_email: userEmail,
      user_name: firstName,
      user_email: userEmail,
    };

    const response = await emailjs.send(
      EMAILJS_CONFIG.SERVICE_ID,
      'welcome_template_id', // You can create a separate welcome email template
      templateParams
    );

    console.log('Welcome email sent successfully:', response);
    return { success: true, message: 'Welcome email sent!' };
  } catch (error) {
    console.error('Failed to send welcome email:', error);
    return { success: false, message: 'Failed to send welcome email' };
  }
};

// For development/testing purposes, we'll also provide a fallback
export const sendMockVerificationEmail = async (userEmail) => {
  // Simulate email sending delay
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // In a real app, you'd store the verification token in a database
  const verificationToken = Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  
  console.log(`Mock verification email sent to: ${userEmail}`);
  console.log(`Verification token: ${verificationToken}`);
  
  return { 
    success: true, 
    message: 'Verification email sent! (Mock mode)',
    token: verificationToken 
  };
}; 