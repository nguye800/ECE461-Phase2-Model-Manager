'use client'

import { useState } from 'react'
import { UploadCloud, Trash, RefreshCw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  resetRegistry,
  uploadArtifact,
  type UploadOptions,
} from '@/lib/registry-api'
import { cn } from '@/lib/utils'

type UploadTabProps = {
  token?: string
}

type Status = { message: string; tone: 'idle' | 'success' | 'error' | 'loading' }

const initialStatus: Status = { message: '', tone: 'idle' }

export function UploadTab({ token }: UploadTabProps) {
  const [mode, setMode] = useState<UploadOptions['mode']>('new')
  const [sourceType, setSourceType] = useState<UploadOptions['sourceType']>('url')
  const [artifactType, setArtifactType] = useState('model')
  const [artifactId, setArtifactId] = useState('')
  const [sourceUrl, setSourceUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [status, setStatus] = useState<Status>(initialStatus)
  const [resetStatus, setResetStatus] = useState<Status>(initialStatus)

  const busy = status.tone === 'loading'

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault()
    const type = artifactType.trim()
    const id = artifactId.trim()

    if (!type || !id) {
      setStatus({ message: 'Artifact type and ID are required.', tone: 'error' })
      return
    }
    if (sourceType === 'url' && !sourceUrl) {
      setStatus({ message: 'Please provide a download URL.', tone: 'error' })
      return
    }
    if (sourceType === 'file' && !file) {
      setStatus({ message: 'Please select a ZIP file to upload.', tone: 'error' })
      return
    }

    setStatus({ message: 'Uploading...', tone: 'loading' })
    const res = await uploadArtifact({
      mode,
      sourceType,
      artifactType: type,
      artifactId: id,
      sourceUrl,
      file,
      token,
    })
    if (res.error) {
      setStatus({ message: res.error, tone: 'error' })
    } else {
      setStatus({ message: 'Upload complete.', tone: 'success' })
      setFile(null)
      if (sourceType === 'url') setSourceUrl('')
    }
  }

  const handleReset = async () => {
    setResetStatus({ message: 'Resetting registry...', tone: 'loading' })
    const res = await resetRegistry(token)
    if (res.error) {
      setResetStatus({ message: res.error, tone: 'error' })
    } else {
      setResetStatus({ message: 'Registry reset to default state.', tone: 'success' })
    }
  }

  return (
    <Card className="shadow-sm">
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div className="space-y-2">
          <CardTitle className="text-2xl">Upload Artifact</CardTitle>
          <CardDescription>
            Register a new artifact or update an existing one via URL or ZIP upload.
          </CardDescription>
        </div>
        <Button
          variant="destructive"
          size="sm"
          onClick={handleReset}
          disabled={resetStatus.tone === 'loading'}
          className="gap-2"
        >
          <Trash className="h-4 w-4" />
          Delete Registry
        </Button>
      </CardHeader>
      <CardContent className="space-y-6">
        <form className="space-y-6" onSubmit={handleUpload}>
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label>Action</Label>
              <RadioGroup
                value={mode}
                onValueChange={(v) => setMode(v as UploadOptions['mode'])}
                className="grid grid-cols-2 gap-3"
              >
                <label
                  htmlFor="mode-new"
                  className={cn(
                    'border-input hover:border-ring bg-card text-card-foreground flex items-center gap-2 rounded-md border px-4 py-3 text-sm shadow-xs transition',
                    mode === 'new' && 'border-ring ring-2 ring-ring/40',
                  )}
                >
                  <RadioGroupItem id="mode-new" value="new" />
                  New artifact
                </label>
                <label
                  htmlFor="mode-update"
                  className={cn(
                    'border-input hover:border-ring bg-card text-card-foreground flex items-center gap-2 rounded-md border px-4 py-3 text-sm shadow-xs transition',
                    mode === 'update' && 'border-ring ring-2 ring-ring/40',
                  )}
                >
                  <RadioGroupItem id="mode-update" value="update" />
                  Update existing
                </label>
              </RadioGroup>
            </div>
            <div className="space-y-2">
              <Label htmlFor="artifact-type">Artifact type</Label>
              <Select value={artifactType} onValueChange={setArtifactType}>
                <SelectTrigger>
                  <SelectValue placeholder="Choose artifact type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="model">Model</SelectItem>
                  <SelectItem value="dataset">Dataset</SelectItem>
                  <SelectItem value="code">Code</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="artifact-id">Artifact ID</Label>
              <Input
                id="artifact-id"
                placeholder="Unique identifier"
                value={artifactId}
                onChange={(e) => setArtifactId(e.target.value)}
                required
                autoComplete="off"
              />
            </div>
            <div className="space-y-2">
              <Label>Source type</Label>
              <Select
                value={sourceType}
                onValueChange={(v) => setSourceType(v as UploadOptions['sourceType'])}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select source" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="url">Downloadable URL</SelectItem>
                  <SelectItem value="file">ZIP file upload</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {sourceType === 'url' ? (
            <div className="space-y-2">
              <Label htmlFor="artifact-url">Download URL</Label>
              <Input
                id="artifact-url"
                type="url"
                placeholder="https://..."
                value={sourceUrl}
                onChange={(e) => setSourceUrl(e.target.value)}
                autoComplete="off"
              />
            </div>
          ) : (
            <div className="space-y-2">
              <Label htmlFor="artifact-zip">ZIP file</Label>
              <Input
                id="artifact-zip"
                type="file"
                accept=".zip"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <p className="text-muted-foreground text-xs">
                The file will be sent as multipart/form-data with the field name &quot;file&quot;.
              </p>
            </div>
          )}

          <div className="flex items-center justify-between gap-4">
            <Button type="submit" className="gap-2" disabled={busy}>
              <UploadCloud className="h-4 w-4" />
              {busy ? 'Uploading...' : mode === 'new' ? 'Upload new artifact' : 'Update artifact'}
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => {
                setArtifactId('')
                setSourceUrl('')
                setFile(null)
                setStatus(initialStatus)
              }}
              className="gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Clear
            </Button>
          </div>
        </form>

        {status.message && (
          <Alert
            variant={status.tone === 'error' ? 'destructive' : 'default'}
            className={status.tone === 'success' ? 'border-green-500/70 text-green-800' : undefined}
          >
            <AlertTitle>
              {status.tone === 'loading'
                ? 'Working'
                : status.tone === 'error'
                  ? 'Error'
                  : 'Status'}
            </AlertTitle>
            <AlertDescription>{status.message}</AlertDescription>
          </Alert>
        )}

        {resetStatus.message && (
          <Alert
            variant={resetStatus.tone === 'error' ? 'destructive' : 'default'}
            className="mt-2"
          >
            <AlertTitle>Registry</AlertTitle>
            <AlertDescription>{resetStatus.message}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}
