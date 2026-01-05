import { useState, useRef, useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import { Send, Bot, User, FileText, Loader2 } from 'lucide-react'
import SourceViewer from './SourceViewer'
import { API_ENDPOINTS, FEATURES } from '../config'

interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  isLoading?: boolean
}

interface Source {
  content: string
  page: number
  source: string
  content_type: string
  relevance_score: number
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedSource, setSelectedSource] = useState<Source | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input,
    }

    setMessages(prev => [...prev, userMessage])
    setInput('')
    setIsLoading(true)

    // Add loading message
    const loadingId = (Date.now() + 1).toString()
    setMessages(prev => [...prev, {
      id: loadingId,
      role: 'assistant',
      content: '',
      isLoading: true,
    }])

    try {
      const response = await fetch(API_ENDPOINTS.query, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: input,
          top_k: 5,
          include_sources: true,
          use_cache: FEATURES.enableCache,
          use_knowledge_graph: FEATURES.enableKnowledgeGraph,
        }),
      })

      const data = await response.json()

      // Replace loading message with actual response
      setMessages(prev => prev.map(msg => 
        msg.id === loadingId
          ? {
              ...msg,
              content: data.answer || 'Sorry, I could not find an answer.',
              sources: data.sources || [],
              isLoading: false,
            }
          : msg
      ))
    } catch (error) {
      setMessages(prev => prev.map(msg =>
        msg.id === loadingId
          ? {
              ...msg,
              content: 'Sorry, there was an error processing your request.',
              isLoading: false,
            }
          : msg
      ))
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="flex gap-6 h-[calc(100vh-12rem)]">
      {/* Chat Area */}
      <div className="flex-1 flex flex-col bg-white/5 rounded-2xl border border-white/10 overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-400/20 to-primary-600/20 flex items-center justify-center mb-4">
                <Bot className="w-8 h-8 text-primary-400" />
              </div>
              <h2 className="text-xl font-semibold mb-2">Ask anything about your documents</h2>
              <p className="text-white/50 max-w-md">
                Upload documents and ask questions. I'll search through your knowledge base
                and provide answers with source citations.
              </p>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex gap-4 animate-slide-up ${
                message.role === 'user' ? 'flex-row-reverse' : ''
              }`}
            >
              <div className={`w-10 h-10 rounded-xl flex items-center justify-center flex-shrink-0 ${
                message.role === 'user'
                  ? 'bg-gradient-to-br from-emerald-400 to-emerald-600'
                  : 'bg-gradient-to-br from-primary-400 to-primary-600'
              }`}>
                {message.role === 'user' ? (
                  <User className="w-5 h-5 text-white" />
                ) : (
                  <Bot className="w-5 h-5 text-white" />
                )}
              </div>

              <div className={`flex-1 max-w-[80%] ${
                message.role === 'user' ? 'text-right' : ''
              }`}>
                <div className={`inline-block rounded-2xl px-5 py-3 ${
                  message.role === 'user'
                    ? 'bg-emerald-500/20 border border-emerald-500/30'
                    : 'bg-white/10 border border-white/10'
                }`}>
                  {message.isLoading ? (
                    <div className="flex items-center gap-2">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-white/60">Thinking...</span>
                    </div>
                  ) : (
                    <div className="prose prose-invert prose-sm max-w-none">
                      <ReactMarkdown>{message.content}</ReactMarkdown>
                    </div>
                  )}
                </div>

                {/* Sources */}
                {message.sources && message.sources.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {message.sources.map((source, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSelectedSource(source)}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 text-xs text-white/60 hover:text-white hover:bg-white/10 transition-all"
                      >
                        <FileText className="w-3 h-3" />
                        <span>Page {source.page}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-white/10">
          <div className="flex gap-3">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-primary-500/50 resize-none"
              rows={1}
            />
            <button
              onClick={sendMessage}
              disabled={!input.trim() || isLoading}
              className="px-5 py-3 bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl text-white font-medium hover:from-primary-400 hover:to-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2"
            >
              <Send className="w-4 h-4" />
              Send
            </button>
          </div>
        </div>
      </div>

      {/* Source Viewer */}
      {selectedSource && (
        <SourceViewer
          source={selectedSource}
          onClose={() => setSelectedSource(null)}
        />
      )}
    </div>
  )
}

