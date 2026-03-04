import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  // State
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState([])
  const [frequentQuestions, setFrequentQuestions] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [showEmpty, setShowEmpty] = useState(true)
  
  const chatBodyRef = useRef(null)
  const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1'

  // Load FAQs on mount
  useEffect(() => {
    loadFrequentQuestions()
  }, [])

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Load Frequent Questions
  const loadFrequentQuestions = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/frequent-questions?limit=12`)
      const data = await response.json()
      
      if (data.success && data.questions) {
        setFrequentQuestions(data.questions)
      } else {
        loadHardcodedFAQs()
      }
    } catch (error) {
      console.error('Failed to load FAQs from API:', error)
      loadHardcodedFAQs()
    }
  }

  // Hardcoded FAQs fallback
  const loadHardcodedFAQs = () => {
    const hardcodedFAQs = [
      {
        question: "Please update dynamic QR code format. The customer details are not being displayed there.",
        category: "Account & Transactions"
      },
      {
        question: "Pls give demo for CKYC AND CERSAI reporting and also let us know the cost to activate the same.",
        category: "Profile Management"
      },
      {
        question: "Please enable WhatsApp service",
        category: "SMS & WhatsApp"
      },
      {
        question: "In Customer app give me Transaction",
        category: "Account & Transactions"
      },
      {
        question: "We need to change the below setting",
        category: "Loan Management"
      },
      {
        question: "RSP option is not enabled in the group option in the profile creation page. Kindly enable the same",
        category: "Profile Management"
      },
      {
        question: "Want to know Televerification process in jainam system",
        category: "Loan Management"
      },
      {
        question: "How to check future receivable of all cases",
        category: "Loan Management"
      },
      {
        question: "EKYC SERVICE REQUIRED",
        category: "Payment & Collection"
      },
      {
        question: "How to take the Print of Confirmation of Accounts of a Particular Ledger statement.",
        category: "Account & Transactions"
      },
      {
        question: "HOW TO ADD GRAUNTER IN A LOAN",
        category: "Loan Management"
      },
      {
        question: "I NEED TRAINING FOR ENACH PROCESSING",
        category: "Loan Management"
      }
    ]
    setFrequentQuestions(hardcodedFAQs)
  }

  // Handle FAQ click
  const handleQuestionClick = async (questionText) => {
    setQuery(questionText)
    
    // Scroll to chat area
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
    
    // Small delay then submit
    setTimeout(() => {
      handleSubmit(new Event('submit'), questionText)
    }, 300)
  }

  // Handle form submit
  const handleSubmit = async (e, questionOverride = null) => {
    e.preventDefault()
    
    const questionText = questionOverride || query.trim()
    if (!questionText || loading) return

    setShowEmpty(false)
    setError(null)
    
    // Add user message
    const userMessage = { type: 'user', text: questionText }
    setMessages(prev => [...prev, userMessage])
    setQuery('')

    // Add typing indicator
    const typingMessage = { type: 'typing', id: 'typing' }
    setMessages(prev => [...prev, typingMessage])
    setLoading(true)

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: questionText,
          user_id: 'web-user',
          channel: 'web'
        })
      })

      const data = await response.json()
      
      // Remove typing indicator
      setMessages(prev => prev.filter(m => m.id !== 'typing'))

      // Add bot response
      const botMessage = {
        type: 'bot',
        text: data.success ? data.answer : (data.fallback_message || 'Sorry, I could not find an answer.'),
        confidence: data.confidence_level,
        relatedQuestions: data.related_questions || []
      }
      setMessages(prev => [...prev, botMessage])

    } catch (error) {
      // Remove typing indicator
      setMessages(prev => prev.filter(m => m.id !== 'typing'))
      
      setError('Failed to connect to the server. Please try again.')
      console.error('Query error:', error)
    } finally {
      setLoading(false)
    }
  }

  // Scroll to bottom
  const scrollToBottom = () => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTop = chatBodyRef.current.scrollHeight
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-container">
          <a href="https://jainamsoftware.com" className="logo">
            <div className="logo-icon">J</div>
            <span>Jainam Software</span>
          </a>
          <nav className="header-nav">
            <a href="https://jainamsoftware.com" className="nav-link">Home</a>
            <a href="https://jainamsoftware.com/products" className="nav-link">Products</a>
            <a href="https://jainamsoftware.com/support" className="nav-link">Support</a>
            <a href="https://jainamsoftware.com" className="website-btn">Visit Website</a>
          </nav>
        </div>
      </header>

      {/* Main Container */}
      <div className="container">
        {/* Hero */}
        <div className="hero">
          <h1>How can we help you today?</h1>
          <p>Get instant answers to your questions with our AI-powered support assistant</p>
        </div>

        {/* Chat Interface */}
        <div className="chat-container">
          <div className="chat-header">
            <h2>Support Assistant</h2>
            <p>Ask anything about Jainam Software products and services</p>
          </div>

          <div className="chat-body" ref={chatBodyRef}>
            {/* Empty State */}
            {showEmpty && messages.length === 0 && (
              <div className="empty-state">
                <div className="empty-state-icon">💬</div>
                <h3>Start a conversation</h3>
                <p>Type your question below or choose from popular topics</p>
              </div>
            )}

            {/* Messages */}
            <div className="messages">
              {messages.map((message, index) => (
                <div key={index} className={`message message-${message.type}`}>
                  {message.type === 'user' && (
                    <div className="message-bubble">{message.text}</div>
                  )}
                  
                  {message.type === 'bot' && (
                    <>
                      <div className="bot-avatar">J</div>
                      <div style={{ flex: 1 }}>
                        <div className="message-bubble">
                          {message.text}
                          {message.confidence && (
                            <span className={`confidence-badge confidence-${message.confidence.toLowerCase()}`}>
                              {message.confidence} confidence
                            </span>
                          )}
                        </div>
                        
                        {message.relatedQuestions && message.relatedQuestions.length > 0 && (
                          <div className="related-section">
                            <div className="related-title">Related questions:</div>
                            <div className="related-chips">
                              {message.relatedQuestions.map((q, i) => (
                                <span 
                                  key={i} 
                                  className="chip" 
                                  onClick={() => handleQuestionClick(q)}
                                >
                                  {q}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </>
                  )}
                  
                  {message.type === 'typing' && (
                    <>
                      <div className="bot-avatar">J</div>
                      <div className="message-bubble typing-indicator">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                      </div>
                    </>
                  )}
                </div>
              ))}

              {/* Error Message */}
              {error && (
                <div className="error-message">{error}</div>
              )}
            </div>
          </div>

          <div className="chat-input">
            <form className="input-wrapper" onSubmit={handleSubmit}>
              <input 
                type="text" 
                className="input-field" 
                placeholder="Type your question here..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={loading}
                autoComplete="off"
              />
              <button 
                type="submit" 
                className="send-btn" 
                disabled={loading || !query.trim()}
              >
                {loading ? 'Sending...' : 'Send'}
              </button>
            </form>
          </div>
        </div>

        {/* FAQ Section */}
        <div className="faq-section">
          <div className="faq-header">
            <h3>Popular Questions</h3>
            <p>Quick answers to commonly asked questions</p>
          </div>
          <div className="faq-grid">
            {frequentQuestions.map((q, index) => (
              <div 
                key={index} 
                className="faq-card" 
                onClick={() => handleQuestionClick(q.question)}
              >
                <div className="faq-category">{q.category || 'General'}</div>
                <div className="faq-question">{q.question}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-links">
          <a href="https://jainamsoftware.com/privacy" className="footer-link">Privacy Policy</a>
          <a href="https://jainamsoftware.com/terms" className="footer-link">Terms of Service</a>
          <a href="https://jainamsoftware.com/contact" className="footer-link">Contact Us</a>
        </div>
        <p>&copy; 2026 Jainam Software. All rights reserved.</p>
      </footer>
    </div>
  )
}

export default App
