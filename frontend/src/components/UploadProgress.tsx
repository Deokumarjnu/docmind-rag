import { useEffect, useState } from 'react'
import { Loader2, CheckCircle, XCircle } from 'lucide-react'
import { API_ENDPOINTS } from '../config'

interface UploadProgressProps {
  taskId: string
  filename: string
  onComplete?: (result: any) => void
  onError?: (error: string) => void
}

interface ProgressStatus {
  status: string
  progress: number
  current_page?: number
  total_pages?: number
  message?: string
  result?: any
  error?: string
}

export default function UploadProgress({
  taskId,
  filename,
  onComplete,
  onError,
}: UploadProgressProps) {
  const [status, setStatus] = useState<ProgressStatus>({
    status: 'pending',
    progress: 0,
  })

  useEffect(() => {
    let intervalId: NodeJS.Timer

    const pollStatus = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.uploadStatus(taskId))
        const data: ProgressStatus = await response.json()
        setStatus(data)

        if (data.status === 'completed') {
          clearInterval(intervalId)
          onComplete?.(data.result)
        } else if (data.status === 'failed') {
          clearInterval(intervalId)
          onError?.(data.error || 'Processing failed')
        }
      } catch {
        // Retry on error
      }
    }

    // Initial poll
    pollStatus()

    // Set up interval
    intervalId = setInterval(pollStatus, 2000)

    return () => clearInterval(intervalId)
  }, [taskId, onComplete, onError])

  const getStatusColor = () => {
    switch (status.status) {
      case 'completed':
        return 'text-emerald-400'
      case 'failed':
        return 'text-red-400'
      case 'processing':
        return 'text-primary-400'
      default:
        return 'text-white/50'
    }
  }

  const getStatusIcon = () => {
    switch (status.status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-emerald-400" />
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />
      default:
        return <Loader2 className="w-5 h-5 animate-spin text-primary-400" />
    }
  }

  return (
    <div className="p-4 bg-white/5 rounded-xl border border-white/10">
      <div className="flex items-center gap-3 mb-3">
        {getStatusIcon()}
        <div className="flex-1 min-w-0">
          <p className="font-medium truncate">{filename}</p>
          <p className={`text-sm ${getStatusColor()}`}>
            {status.message || status.status}
          </p>
        </div>
      </div>

      {/* Progress Bar */}
      {(status.status === 'processing' || status.status === 'pending') && (
        <div className="space-y-2">
          <div className="h-2 bg-white/10 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary-400 to-primary-600 transition-all duration-300"
              style={{ width: `${status.progress}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-white/50">
            <span>
              {status.current_page !== undefined && status.total_pages !== undefined
                ? `Page ${status.current_page} of ${status.total_pages}`
                : 'Processing...'}
            </span>
            <span>{status.progress}%</span>
          </div>
        </div>
      )}

      {/* Result */}
      {status.status === 'completed' && status.result && (
        <div className="text-sm text-white/60 mt-2">
          <span className="text-emerald-400">{status.result.total_chunks}</span> chunks indexed
        </div>
      )}

      {/* Error */}
      {status.status === 'failed' && status.error && (
        <p className="text-sm text-red-400 mt-2">{status.error}</p>
      )}
    </div>
  )
}

