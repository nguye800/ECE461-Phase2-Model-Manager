'use client'

import { useMemo, useState } from 'react'
import {
  ArrowRight,
  ChevronDown,
  ChevronUp,
  DollarSign,
  Download,
  FileSearch,
  Gavel,
  Info,
  RefreshCw,
  Trash2,
} from 'lucide-react'
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from '@/components/ui/accordion'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Pagination, PaginationContent, PaginationItem, PaginationLink } from '@/components/ui/pagination'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  checkLicense,
  deleteArtifact,
  downloadArtifact,
  fetchAudits,
  fetchCost,
  fetchLineage,
  fetchRatings,
  searchArtifacts,
  type ArtifactSummary,
} from '@/lib/registry-api'
import { cn } from '@/lib/utils'

type SearchTabProps = {
  token?: string
}

type DetailState = {
  loading: boolean
  error?: string
  ratings?: unknown
  cost?: unknown
  lineage?: unknown
  license?: unknown
  audits?: unknown
  downloadUrl?: string
}

type Status = { message: string; tone: 'idle' | 'error' | 'success' }

const RESULTS_PER_PAGE = 10

export function SearchTab({ token }: SearchTabProps) {
  const [query, setQuery] = useState('')
  const [defaultType, setDefaultType] = useState('model')
  const [results, setResults] = useState<ArtifactSummary[]>([])
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState<Status>({ message: '', tone: 'idle' })
  const [page, setPage] = useState(1)
  const [detailState, setDetailState] = useState<Record<string, DetailState>>({})

  const paged = useMemo(() => {
    const start = (page - 1) * RESULTS_PER_PAGE
    return results.slice(start, start + RESULTS_PER_PAGE)
  }, [page, results])

  const totalPages = Math.max(1, Math.ceil(results.length / RESULTS_PER_PAGE))

  const reset = () => {
    setResults([])
    setPage(1)
    setDetailState({})
    setStatus({ message: '', tone: 'idle' })
  }

  const doSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    const normalized = query.trim() || '*'
    setLoading(true)
    setStatus({ message: '', tone: 'idle' })
    setDetailState({})
    setPage(1)

    const res = await searchArtifacts(normalized, token, defaultType)
    setLoading(false)

    if (res.error) {
      setStatus({ message: res.error, tone: 'error' })
      setResults([])
      return
    }
    const artifacts = res.data?.artifacts ?? []
    setResults(artifacts)
    setStatus(
      artifacts.length
        ? { message: `Found ${artifacts.length} artifact(s).`, tone: 'success' }
        : { message: 'No Results Found', tone: 'error' },
    )
  }

  const loadDetails = async (artifact: ArtifactSummary) => {
    const key = artifact.id
    setDetailState((prev) => ({
      ...prev,
      [key]: { ...(prev[key] ?? {}), loading: true, error: undefined },
    }))

    const [rateRes, lineageRes] = await Promise.all([
      fetchRatings(artifact.id, token),
      fetchLineage(artifact.id, token),
    ])

    setDetailState((prev) => ({
      ...prev,
      [key]: {
        ...(prev[key] ?? {}),
        loading: false,
        ratings: rateRes.data,
        lineage: lineageRes.data,
        error: rateRes.error || lineageRes.error,
      },
    }))
  }

  const handleDelete = async (artifact: ArtifactSummary) => {
    if (!artifact.type) {
      setStatus({ message: 'Artifact type is required to delete a model.', tone: 'error' })
      return
    }
    const res = await deleteArtifact(artifact.type, artifact.id, token)
    if (res.error) {
      setStatus({ message: res.error, tone: 'error' })
    } else {
      setResults((prev) => prev.filter((r) => r.id !== artifact.id))
      setStatus({ message: `Deleted ${artifact.name || artifact.id}.`, tone: 'success' })
    }
  }

  const handleDownload = async (artifact: ArtifactSummary) => {
    if (!artifact.type) {
      setStatus({ message: 'Artifact type is required to download a model.', tone: 'error' })
      return
    }
    const res = await downloadArtifact(artifact.type, artifact.id, token)
    if (res.error) {
      setStatus({ message: res.error, tone: 'error' })
      return
    }
    const url = res.data?.url ?? res.data?.downloadUrl
    if (url) {
      setDetailState((prev) => ({
        ...prev,
        [artifact.id]: { ...(prev[artifact.id] ?? {}), downloadUrl: url },
      }))
      window.open(url, '_blank')
      setStatus({ message: 'Opened download URL in a new tab.', tone: 'success' })
    } else {
      setStatus({ message: 'Download URL not returned by the API.', tone: 'error' })
    }
  }

  const handleCostPopup = async (artifact: ArtifactSummary) => {
    if (!artifact.type) {
      setStatus({ message: 'Artifact type is required to fetch cost.', tone: 'error' })
      return
    }
    const res = await fetchCost(artifact.type, artifact.id, token)
    setDetailState((prev) => ({
      ...prev,
      [artifact.id]: { ...(prev[artifact.id] ?? {}), cost: res.data, error: res.error },
    }))
    if (res.error) {
      setStatus({ message: res.error, tone: 'error' })
      return
    }
    const payload = JSON.stringify(res.data ?? {}, null, 2)
    window.alert(`Cost for ${artifact.name || artifact.id}:\n${payload}`)
  }

  const handleLicense = async (artifact: ArtifactSummary) => {
    const res = await checkLicense(artifact.id, token)
    setDetailState((prev) => ({
      ...prev,
      [artifact.id]: { ...(prev[artifact.id] ?? {}), license: res.data, error: res.error },
    }))
    if (res.error) setStatus({ message: res.error, tone: 'error' })
  }

  const handleAudits = async (artifact: ArtifactSummary) => {
    if (!artifact.type) {
      setStatus({ message: 'Artifact type is required to fetch audits.', tone: 'error' })
      return
    }
    const res = await fetchAudits(artifact.type, artifact.id, token)
    setDetailState((prev) => ({
      ...prev,
      [artifact.id]: { ...(prev[artifact.id] ?? {}), audits: res.data, error: res.error },
    }))
    if (res.error) setStatus({ message: res.error, tone: 'error' })
  }

  return (
    <Card className="shadow-sm">
      <CardHeader>
        <CardTitle className="text-2xl">Search Registry</CardTitle>
        <CardDescription>
          Search the registry, expand results for metadata, and run model actions.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form className="space-y-3" onSubmit={doSearch}>
          <div className="grid gap-3 md:grid-cols-[1fr,220px,140px]">
            <div className="space-y-2">
              <Label htmlFor="search-query">Query</Label>
              <Input
                id="search-query"
                placeholder='e.g. "whisper" or "*" for all'
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') doSearch()
                }}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="default-type">Default artifact type</Label>
              <Select value={defaultType} onValueChange={setDefaultType}>
                <SelectTrigger id="default-type">
                  <SelectValue placeholder="model" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="model">Model</SelectItem>
                  <SelectItem value="dataset">Dataset</SelectItem>
                  <SelectItem value="code">Code</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end gap-2">
              <Button type="submit" className="w-full gap-2" disabled={loading}>
                <FileSearch className="h-4 w-4" />
                {loading ? 'Searching...' : 'Search'}
              </Button>
              <Button type="button" variant="outline" size="icon" onClick={reset}>
                <RefreshCw className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </form>

        {status.message && (
          <Alert variant={status.tone === 'error' ? 'destructive' : 'default'}>
            <AlertTitle>{status.tone === 'error' ? 'Notice' : 'Status'}</AlertTitle>
            <AlertDescription>{status.message}</AlertDescription>
          </Alert>
        )}

        {results.length > 0 && (
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>Showing {Math.min(results.length, page * RESULTS_PER_PAGE)} of {results.length}</span>
            <Badge variant="outline">Page {page}</Badge>
          </div>
        )}

        <div className="space-y-3">
          {paged.length > 0 ? (
            paged.map((artifact) => (
              <Card key={artifact.id}>
                <CardHeader className="flex flex-row items-start justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <CardTitle className="text-lg">{artifact.name || artifact.id}</CardTitle>
                      <Badge variant="secondary">{artifact.type || defaultType}</Badge>
                    </div>
                    <CardDescription className="text-xs font-mono text-muted-foreground">
                      id: {artifact.id}
                    </CardDescription>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Button variant="outline" size="sm" className="gap-1" onClick={() => handleDownload(artifact)}>
                      <Download className="h-4 w-4" />
                      Download
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      onClick={() => handleLicense(artifact)}
                    >
                      <Gavel className="h-4 w-4" />
                      License
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      onClick={() => handleAudits(artifact)}
                    >
                      <Info className="h-4 w-4" />
                      Audits
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      className="gap-1"
                      onClick={() => handleCostPopup(artifact)}
                    >
                      <DollarSign className="h-4 w-4" />
                      See Cost
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      className="gap-1"
                      onClick={() => handleDelete(artifact)}
                    >
                      <Trash2 className="h-4 w-4" />
                      Delete Model
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="space-y-2">
                  <Accordion
                    type="single"
                    collapsible
                    onValueChange={(val) => {
                      if (val === artifact.id) loadDetails(artifact)
                    }}
                  >
                    <AccordionItem value={artifact.id}>
                      <AccordionTrigger className="text-sm">
                        <div className="flex items-center gap-2">
                          Details
                          <ArrowRight className="h-3 w-3" />
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <DetailSection
                          state={detailState[artifact.id]}
                          onRefresh={() => loadDetails(artifact)}
                          artifact={artifact}
                        />
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </CardContent>
              </Card>
            ))
          ) : (
            !loading && (
              <div className="text-center text-sm text-muted-foreground">
                No results yet. Run a search to see artifacts.
              </div>
            )
          )}
        </div>

        {results.length > RESULTS_PER_PAGE && (
          <Pagination>
            <PaginationContent>
              {Array.from({ length: totalPages }).map((_, idx) => {
                const pageNumber = idx + 1
                return (
                  <PaginationItem key={pageNumber}>
                    <PaginationLink
                      href="#"
                      isActive={pageNumber === page}
                      onClick={(e) => {
                        e.preventDefault()
                        setPage(pageNumber)
                      }}
                    >
                      {pageNumber}
                    </PaginationLink>
                  </PaginationItem>
                )
              })}
            </PaginationContent>
          </Pagination>
        )}
      </CardContent>
    </Card>
  )
}

type DetailSectionProps = {
  state?: DetailState
  onRefresh: () => void
  artifact: ArtifactSummary
}

function DetailSection({ state, onRefresh, artifact }: DetailSectionProps) {
  if (!state) {
    return (
      <div className="text-sm text-muted-foreground">
        Expand to fetch ratings and lineage. Use “See Cost” to retrieve cost details.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm text-muted-foreground">
          {state.loading ? 'Loading details...' : 'Fetched details'}
        </span>
        <Button variant="ghost" size="sm" className="gap-1" onClick={onRefresh}>
          <RefreshCw className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {state.error && (
        <Alert variant="destructive">
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{state.error}</AlertDescription>
        </Alert>
      )}

      <div className="grid gap-3 md:grid-cols-2">
        <InfoBlock title="Ratings" data={state.ratings} />
        <InfoBlock title="Cost" data={state.cost ?? 'Use the See Cost button to fetch cost.'} />
      </div>
      <InfoBlock title="Lineage" data={state.lineage} />
      <InfoBlock title="License Check" data={state.license} />
      <InfoBlock title="Audits" data={state.audits} />
      <InfoBlock title="Download URL" data={state.downloadUrl || 'Use the download button above.'} />

      <div className="rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-2 pb-1 font-semibold">
          <ChevronDown className="h-4 w-4" />
          Actions called
        </div>
        <ul className="grid gap-1 md:grid-cols-2">
          <li className="flex items-center gap-1">
            <ChevronUp className="h-3 w-3" /> GET /artifact/model/{artifact.id}/rate
          </li>
          <li className="flex items-center gap-1">
            <ChevronUp className="h-3 w-3" />{' '}
            {artifact.type
              ? `GET /artifact/${artifact.type}/${artifact.id}/cost`
              : 'Cost requires artifact type'}
          </li>
          <li className="flex items-center gap-1">
            <ChevronUp className="h-3 w-3" /> GET /artifact/model/{artifact.id}/lineage
          </li>
        </ul>
      </div>
    </div>
  )
}

type InfoBlockProps = {
  title: string
  data: unknown
}

function InfoBlock({ title, data }: InfoBlockProps) {
  const content =
    data === undefined || data === null
      ? 'N/A'
      : typeof data === 'string'
        ? data
        : JSON.stringify(data, null, 2)

  return (
    <div className="rounded-md border bg-card/40 p-3">
      <div className="flex items-center gap-2 pb-1 text-sm font-semibold">
        <Info className="h-4 w-4 text-muted-foreground" />
        {title}
      </div>
      <ScrollArea className="h-auto max-h-56 rounded border bg-muted/30 p-2 text-xs font-mono">
        <pre className="whitespace-pre-wrap">{content}</pre>
      </ScrollArea>
    </div>
  )
}
