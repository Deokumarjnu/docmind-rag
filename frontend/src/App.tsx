import { useState } from 'react'
import ChatInterface from './components/ChatInterface'
import FileUpload from './components/FileUpload'
import DocumentList from './components/DocumentList'
import { FileText, MessageSquare, Upload, Zap } from 'lucide-react'

type Tab = 'chat' | 'upload' | 'documents'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('chat')

  return (
    <div className="min-h-screen text-white">
      {/* Header */}
      <header className="border-b border-white/10 backdrop-blur-sm bg-white/5">
        <div className="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-400 to-primary-600 flex items-center justify-center animate-pulse-glow">
              <Zap className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">DocMind RAG</h1>
              <p className="text-xs text-white/50">Enterprise Document Intelligence</p>
            </div>
          </div>
          
          {/* Navigation */}
          <nav className="flex gap-1 bg-white/5 p-1 rounded-xl">
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'chat'
                  ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                  : 'text-white/60 hover:text-white hover:bg-white/10'
              }`}
            >
              <MessageSquare className="w-4 h-4" />
              Chat
            </button>
            <button
              onClick={() => setActiveTab('upload')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'upload'
                  ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                  : 'text-white/60 hover:text-white hover:bg-white/10'
              }`}
            >
              <Upload className="w-4 h-4" />
              Upload
            </button>
            <button
              onClick={() => setActiveTab('documents')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === 'documents'
                  ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                  : 'text-white/60 hover:text-white hover:bg-white/10'
              }`}
            >
              <FileText className="w-4 h-4" />
              Documents
            </button>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-8">
        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'upload' && <FileUpload />}
        {activeTab === 'documents' && <DocumentList />}
      </main>
    </div>
  )
}

export default App

