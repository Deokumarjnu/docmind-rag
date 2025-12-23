import { X, FileText, Hash, BarChart3 } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Source {
  content: string
  page: number
  source: string
  content_type: string
  relevance_score: number
}

interface SourceViewerProps {
  source: Source
  onClose: () => void
}

export default function SourceViewer({ source, onClose }: SourceViewerProps) {
  const getContentTypeIcon = (type: string) => {
    switch (type) {
      case 'table':
        return <BarChart3 className="w-4 h-4" />
      case 'code':
        return <Hash className="w-4 h-4" />
      default:
        return <FileText className="w-4 h-4" />
    }
  }

  const getContentTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      text: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
      table: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
      code: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
      chart: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
      handwriting: 'bg-pink-500/20 text-pink-400 border-pink-500/30',
    }
    return colors[type] || 'bg-white/10 text-white/60 border-white/20'
  }

  const relevancePercent = Math.round(source.relevance_score * 100)

  return (
    <div className="w-96 bg-white/5 rounded-2xl border border-white/10 overflow-hidden flex flex-col animate-slide-up">
      {/* Header */}
      <div className="p-4 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary-500/20 flex items-center justify-center">
            {getContentTypeIcon(source.content_type)}
          </div>
          <div>
            <h3 className="font-medium text-sm">Source Document</h3>
            <p className="text-xs text-white/50">Page {source.page}</p>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 rounded-lg hover:bg-white/10 transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Metadata */}
      <div className="p-4 border-b border-white/10 space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/50">Source</span>
          <span className="font-mono text-xs truncate max-w-[200px]">
            {source.source.split('/').pop()}
          </span>
        </div>
        
        <div className="flex items-center justify-between text-sm">
          <span className="text-white/50">Content Type</span>
          <span className={`px-2 py-0.5 rounded text-xs font-medium border ${getContentTypeColor(source.content_type)}`}>
            {source.content_type}
          </span>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-white/50">Relevance</span>
          <div className="flex items-center gap-2">
            <div className="w-20 h-1.5 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary-400 to-primary-600"
                style={{ width: `${relevancePercent}%` }}
              />
            </div>
            <span className="text-xs font-medium">{relevancePercent}%</span>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="prose prose-invert prose-sm max-w-none">
          <ReactMarkdown>{source.content}</ReactMarkdown>
        </div>
      </div>
    </div>
  )
}

