import { useState, useEffect } from 'react'
import { FileText, Trash2, RefreshCw, Loader2, AlertCircle } from 'lucide-react'
import { API_ENDPOINTS } from '../config'

interface Document {
  document_id: string
  filename: string
  total_chunks: number
  pages: number
  content_types: string[]
  language?: string
}

export default function DocumentList() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const fetchDocuments = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const response = await fetch(API_ENDPOINTS.documents)
      const data = await response.json()
      setDocuments(data.documents || [])
    } catch (err) {
      setError('Failed to load documents')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchDocuments()
  }, [])

  const deleteDocument = async (documentId: string) => {
    if (!confirm('Are you sure you want to delete this document?')) return

    setDeletingId(documentId)
    
    try {
      await fetch(API_ENDPOINTS.document(documentId), {
        method: 'DELETE',
      })
      setDocuments(prev => prev.filter(d => d.document_id !== documentId))
    } catch {
      setError('Failed to delete document')
    } finally {
      setDeletingId(null)
    }
  }

  const getContentTypeColor = (type: string) => {
    const colors: Record<string, string> = {
      text: 'bg-blue-500/20 text-blue-400',
      table: 'bg-emerald-500/20 text-emerald-400',
      code: 'bg-amber-500/20 text-amber-400',
      chart: 'bg-purple-500/20 text-purple-400',
      handwriting: 'bg-pink-500/20 text-pink-400',
      image: 'bg-cyan-500/20 text-cyan-400',
    }
    return colors[type] || 'bg-white/10 text-white/60'
  }

  return (
    <div className="max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold">Documents</h2>
          <p className="text-white/50 text-sm mt-1">
            {documents.length} document{documents.length !== 1 ? 's' : ''} in your knowledge base
          </p>
        </div>
        <button
          onClick={fetchDocuments}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 bg-white/10 hover:bg-white/20 rounded-xl transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Error State */}
      {error && (
        <div className="mb-6 p-4 bg-red-500/20 border border-red-500/30 rounded-xl flex items-center gap-3">
          <AlertCircle className="w-5 h-5 text-red-400" />
          <span className="text-red-400">{error}</span>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin text-primary-400" />
        </div>
      )}

      {/* Empty State */}
      {!isLoading && documents.length === 0 && (
        <div className="text-center py-12 bg-white/5 rounded-2xl border border-white/10">
          <div className="w-16 h-16 rounded-2xl bg-white/10 flex items-center justify-center mx-auto mb-4">
            <FileText className="w-8 h-8 text-white/40" />
          </div>
          <h3 className="text-lg font-medium mb-2">No documents yet</h3>
          <p className="text-white/50">
            Upload some documents to get started with your knowledge base.
          </p>
        </div>
      )}

      {/* Document List */}
      {!isLoading && documents.length > 0 && (
        <div className="bg-white/5 rounded-2xl border border-white/10 overflow-hidden">
          <div className="divide-y divide-white/10">
            {documents.map((doc) => (
              <div
                key={doc.document_id}
                className="p-5 hover:bg-white/5 transition-colors"
              >
                <div className="flex items-start gap-4">
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500/20 to-primary-600/20 flex items-center justify-center flex-shrink-0">
                    <FileText className="w-6 h-6 text-primary-400" />
                  </div>

                  <div className="flex-1 min-w-0">
                    <h3 className="font-medium truncate">{doc.filename}</h3>
                    
                    <div className="flex items-center gap-4 mt-2 text-sm text-white/50">
                      <span>{doc.pages} pages</span>
                      <span>•</span>
                      <span>{doc.total_chunks} chunks</span>
                      {doc.language && (
                        <>
                          <span>•</span>
                          <span>{doc.language.toUpperCase()}</span>
                        </>
                      )}
                    </div>

                    {/* Content Types */}
                    <div className="flex flex-wrap gap-2 mt-3">
                      {doc.content_types.map((type) => (
                        <span
                          key={type}
                          className={`px-2 py-0.5 rounded-md text-xs font-medium ${getContentTypeColor(type)}`}
                        >
                          {type}
                        </span>
                      ))}
                    </div>
                  </div>

                  {/* Delete Button */}
                  <button
                    onClick={() => deleteDocument(doc.document_id)}
                    disabled={deletingId === doc.document_id}
                    className="p-2 rounded-lg text-white/40 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
                  >
                    {deletingId === doc.document_id ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Trash2 className="w-5 h-5" />
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

