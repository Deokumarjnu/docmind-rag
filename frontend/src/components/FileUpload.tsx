import { useState, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, File, CheckCircle, XCircle, Loader2, FileText } from 'lucide-react'
import UploadProgress from './UploadProgress'

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

      const response = await fetch('/api/upload', {
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
          ? { ...u, taskId: data.task_id, status: 'processing', progress: 10 }
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
        const response = await fetch(`/api/upload/status/${taskId}`)
        const data = await response.json()

        setUploads(prev => prev.map(u => {
          if (u.file !== file) return u

          if (data.status === 'completed') {
            return {
              ...u,
              status: 'completed',
              progress: 100,
              result: data.result,
            }
          } else if (data.status === 'failed') {
            return {
              ...u,
              status: 'failed',
              error: data.error || 'Processing failed',
            }
          } else {
            return {
              ...u,
              status: 'processing',
              progress: data.progress || u.progress,
            }
          }
        }))

        // Continue polling if still processing
        if (data.status === 'processing' || data.status === 'pending') {
          setTimeout(poll, 2000)
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
      'text/html': ['.html'],
      'text/plain': ['.txt'],
    },
    multiple: true,
  })

  return (
    <div className="max-w-3xl mx-auto space-y-6">
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
              or click to browse • PDF, DOCX, HTML, TXT supported
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
        <h3 className="font-medium mb-4">Processing Features</h3>
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
      </div>
    </div>
  )
}

