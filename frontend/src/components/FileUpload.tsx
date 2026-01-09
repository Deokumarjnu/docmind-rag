import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, CheckCircle, XCircle, Loader2, FileText, Zap } from 'lucide-react'
import UploadProgress from './UploadProgress'
import { API_ENDPOINTS } from '../config'

interface UploadState {
  file: File
  taskId?: string
  status: 'pending' | 'uploading' | 'processing' | 'completed' | 'failed'
  progress: number
  error?: string
  result?: {
    total_chunks: number
    document_id: string
  }
}

export default function FileUpload() {
  const [uploads, setUploads] = useState<UploadState[]>([])
  const [fastMode, setFastMode] = useState(true) // Default to fast mode for quicker uploads

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const newUploads: UploadState[] = acceptedFiles.map(file => ({
      file,
      status: 'pending',
      progress: 0,
    }))

    setUploads(prev => [...prev, ...newUploads])

    // Upload each file
    for (const upload of newUploads) {
      await uploadFile(upload.file)
    }
  }, [])

  const uploadFile = async (file: File) => {
    // Update status to uploading
    setUploads(prev => prev.map(u => 
      u.file === file ? { ...u, status: 'uploading' } : u
    ))

    try {
      const formData = new FormData()
      formData.append('file', file)

      // Use fast mode (no deep agents) for quicker processing
      const url = fastMode ? `${API_ENDPOINTS.upload}?use_deep_agents=false` : API_ENDPOINTS.upload
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Upload failed')
      }

      const data = await response.json()

      // Update with task ID and start polling
      setUploads(prev => prev.map(u =>
        u.file === file
          ? { ...u, taskId: data.task_id, status: 'processing', progress: 5 }
          : u
      ))

      // Poll for progress
      pollProgress(file, data.task_id)

    } catch (error) {
      setUploads(prev => prev.map(u =>
        u.file === file
          ? { ...u, status: 'failed', error: 'Upload failed' }
          : u
      ))
    }
  }

  const pollProgress = async (file: File, taskId: string) => {
    const poll = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.uploadStatus(taskId))
        const data = await response.json()

        setUploads(prev => prev.map(u => {
          if (u.file !== file) return u

          if (data.status === 'completed' || data.status === 'SUCCESS') {
            return {
              ...u,
              status: 'completed',
              progress: 100,
              result: data.result,
            }
          } else if (data.status === 'failed' || data.status === 'FAILURE') {
            return {
              ...u,
              status: 'failed',
              error: data.error || 'Processing failed',
            }
          } else {
            // Use progress from backend, with minimum of current progress
            const newProgress = Math.max(data.progress || 0, u.progress)
            return {
              ...u,
              status: 'processing',
              progress: newProgress,
            }
          }
        }))

        // Continue polling if still processing
        if (data.status !== 'completed' && data.status !== 'failed' && 
            data.status !== 'SUCCESS' && data.status !== 'FAILURE') {
          setTimeout(poll, 1500)
        }
      } catch {
        // Retry on error
        setTimeout(poll, 3000)
      }
    }

    poll()
  }

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls'],
      'text/csv': ['.csv'],
      'application/json': ['.json', '.jsonl'],
      'text/markdown': ['.md', '.markdown'],
      'text/html': ['.html'],
      'text/plain': ['.txt'],
    },
    multiple: true,
  })

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Fast Mode Toggle */}
      <div className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/10">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${fastMode ? 'bg-amber-500/20' : 'bg-white/10'}`}>
            <Zap className={`w-5 h-5 ${fastMode ? 'text-amber-400' : 'text-white/40'}`} />
          </div>
          <div>
            <h4 className="font-medium">Fast Mode</h4>
            <p className="text-xs text-white/50">
              {fastMode ? 'Quick processing (~10s)' : 'Deep Agent processing (~60-120s)'}
            </p>
          </div>
        </div>
        <button
          onClick={() => setFastMode(!fastMode)}
          className={`relative w-12 h-6 rounded-full transition-colors ${fastMode ? 'bg-amber-500' : 'bg-white/20'}`}
        >
          <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${fastMode ? 'left-7' : 'left-1'}`} />
        </button>
      </div>

      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
          isDragActive
            ? 'border-primary-500 bg-primary-500/10'
            : 'border-white/20 hover:border-white/40 bg-white/5'
        }`}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center gap-4">
          <div className={`w-16 h-16 rounded-2xl flex items-center justify-center transition-colors ${
            isDragActive ? 'bg-primary-500/20' : 'bg-white/10'
          }`}>
            <Upload className={`w-8 h-8 ${isDragActive ? 'text-primary-400' : 'text-white/60'}`} />
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-1">
              {isDragActive ? 'Drop files here' : 'Drag & drop documents'}
            </h3>
            <p className="text-white/50 text-sm">
              or click to browse • PDF, DOCX, XLSX, CSV, JSON, Markdown, HTML, TXT
            </p>
          </div>
        </div>
      </div>

      {/* Upload List */}
      {uploads.length > 0 && (
        <div className="bg-white/5 rounded-2xl border border-white/10 overflow-hidden">
          <div className="p-4 border-b border-white/10">
            <h3 className="font-medium">Uploads</h3>
          </div>
          <div className="divide-y divide-white/10">
            {uploads.map((upload, idx) => (
              <div key={idx} className="p-4 flex items-center gap-4">
                <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
                  <FileText className="w-5 h-5 text-white/60" />
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className="font-medium truncate">{upload.file.name}</p>
                  <div className="flex items-center gap-2 mt-1">
                    {upload.status === 'pending' && (
                      <span className="text-xs text-white/50">Waiting...</span>
                    )}
                    {upload.status === 'uploading' && (
                      <span className="text-xs text-primary-400">Uploading...</span>
                    )}
                    {upload.status === 'processing' && (
                      <>
                        <Loader2 className="w-3 h-3 animate-spin text-primary-400" />
                        <span className="text-xs text-primary-400">Processing {upload.progress}%</span>
                      </>
                    )}
                    {upload.status === 'completed' && (
                      <>
                        <CheckCircle className="w-3 h-3 text-emerald-400" />
                        <span className="text-xs text-emerald-400">
                          Completed • {upload.result?.total_chunks} chunks
                        </span>
                      </>
                    )}
                    {upload.status === 'failed' && (
                      <>
                        <XCircle className="w-3 h-3 text-red-400" />
                        <span className="text-xs text-red-400">{upload.error}</span>
                      </>
                    )}
                  </div>
                </div>

                {/* Progress Bar */}
                {(upload.status === 'uploading' || upload.status === 'processing') && (
                  <div className="w-24">
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-primary-400 to-primary-600 transition-all duration-300"
                        style={{ width: `${upload.progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Info */}
      <div className="bg-white/5 rounded-2xl border border-white/10 p-6">
        <h3 className="font-medium mb-4">
          {fastMode ? 'Fast Mode Features' : 'Deep Agent Features'}
        </h3>
        {fastMode ? (
          <ul className="space-y-3 text-sm text-white/60">
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-amber-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Zap className="w-3 h-3 text-amber-400" />
              </div>
              <span>Quick parallel page processing (~10 seconds)</span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-amber-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Zap className="w-3 h-3 text-amber-400" />
              </div>
              <span>Automatic page type detection (text, table, image)</span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-amber-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <Zap className="w-3 h-3 text-amber-400" />
              </div>
              <span>Content-aware chunking for better retrieval</span>
            </li>
          </ul>
        ) : (
          <ul className="space-y-3 text-sm text-white/60">
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <CheckCircle className="w-3 h-3 text-emerald-400" />
              </div>
              <span>Deep Agent orchestration for intelligent content extraction</span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <CheckCircle className="w-3 h-3 text-emerald-400" />
              </div>
              <span>Vision AI for charts, diagrams, and handwritten text</span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <CheckCircle className="w-3 h-3 text-emerald-400" />
              </div>
              <span>Multi-page table merging and structure preservation</span>
            </li>
            <li className="flex items-start gap-3">
              <div className="w-5 h-5 rounded bg-emerald-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                <CheckCircle className="w-3 h-3 text-emerald-400" />
              </div>
              <span>Code-aware chunking with AST parsing</span>
            </li>
          </ul>
        )}
      </div>
    </div>
  )
}

