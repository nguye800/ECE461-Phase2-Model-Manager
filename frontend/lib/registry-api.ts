'use client'

/**
 * Lightweight client-side helpers for the model registry endpoints.
 * All requests are relative to NEXT_PUBLIC_API_BASE if provided, otherwise same-origin.
 */

export type ArtifactSummary = {
  id: string
  name: string
  type?: string
  [key: string]: any
}

export type FetchResult<T> = { data?: T; error?: string }

// Default to the provided API Gateway base if NEXT_PUBLIC_API_BASE is not set.
const apiBase =
  typeof process !== 'undefined'
    ? (process.env.NEXT_PUBLIC_API_BASE ?? 'https://jlx4q9cg22.execute-api.us-east-1.amazonaws.com').replace(
        /\/+$/,
        '',
      )
    : ''

const withAuthHeaders = (token?: string, extra?: HeadersInit): HeadersInit => ({
  ...(token ? { Authorization: `Bearer ${token}` } : {}),
  ...extra,
})

const buildUrl = (path: string) => `${apiBase}${path}`

const parseError = async (response: Response) => {
  try {
    const body = await response.json()
    const message =
      typeof body === 'string'
        ? body
        : body?.message || body?.error || JSON.stringify(body, null, 2)
    return `${response.status} ${response.statusText}${message ? ` - ${message}` : ''}`
  } catch {
    const text = await response.text().catch(() => '')
    return `${response.status} ${response.statusText}${text ? ` - ${text}` : ''}`
  }
}

const httpJson = async <T>(path: string, init: RequestInit): Promise<FetchResult<T>> => {
  try {
    const res = await fetch(buildUrl(path), init)
    if (!res.ok) {
      return { error: await parseError(res) }
    }
    if (res.status === 204) return { data: undefined }
    const data = (await res.json()) as T
    return { data }
  } catch (err) {
    return { error: err instanceof Error ? err.message : 'Unknown error' }
  }
}

export const authenticate = async (): Promise<FetchResult<{ token?: string }>> =>
  httpJson('/authenticate', { method: 'PUT' })

export const resetRegistry = async (token?: string) =>
  httpJson('/reset', { method: 'DELETE', headers: withAuthHeaders(token) })

export type UploadOptions = {
  mode: 'new' | 'update'
  sourceType: 'url' | 'file'
  artifactType: string
  artifactId: string
  sourceUrl?: string
  file?: File | null
  token?: string
}

export const uploadArtifact = async ({
  mode,
  sourceType,
  artifactType,
  artifactId,
  sourceUrl,
  file,
  token,
}: UploadOptions) => {
  const path =
    mode === 'new'
      ? `/artifacts/${encodeURIComponent(artifactType)}`
      : `/artifacts/${encodeURIComponent(artifactType)}/${encodeURIComponent(artifactId)}`

  if (sourceType === 'file') {
    const form = new FormData()
    if (file) form.append('file', file)
    if (artifactId) form.append('id', artifactId)
    return httpJson(path, {
      method: mode === 'new' ? 'POST' : 'PUT',
      headers: withAuthHeaders(token),
      body: form,
    })
  }

  return httpJson(path, {
    method: mode === 'new' ? 'POST' : 'PUT',
    headers: withAuthHeaders(token, { 'Content-Type': 'application/json' }),
    body: JSON.stringify({
      id: artifactId || undefined,
      sourceUrl: sourceUrl,
    }),
  })
}

const normalizeArtifact = (item: any, fallbackType?: string): ArtifactSummary => {
  const id =
    item?.id ??
    item?.artifact_id ??
    item?.artifactId ??
    item?.pk ??
    item?.name ??
    item?.artifactName ??
    ''
  const name =
    item?.name ?? item?.artifactName ?? item?.title ?? item?.display_name ?? String(id ?? '')
  const type = item?.artifact_type ?? item?.type ?? item?.kind ?? fallbackType
  return { ...(item ?? {}), id: String(id ?? ''), name, type }
}

const extractArtifacts = (payload: any, fallbackType?: string): ArtifactSummary[] => {
  if (!payload) return []
  if (Array.isArray(payload)) return payload.map((item) => normalizeArtifact(item, fallbackType))
  const collections = [
    payload?.artifacts,
    payload?.items,
    payload?.results,
    payload?.data,
    payload?.records,
  ].find(Array.isArray)
  if (Array.isArray(collections)) return collections.map((i) => normalizeArtifact(i, fallbackType))
  // Single object case
  if (typeof payload === 'object') return [normalizeArtifact(payload, fallbackType)]
  return []
}

export const searchArtifacts = async (
  query: string,
  token?: string,
  fallbackType?: string,
): Promise<FetchResult<{ artifacts: ArtifactSummary[]; source: string }>> => {
  const strategies: Array<() => Promise<FetchResult<{ artifacts: ArtifactSummary[] }>>> = [
    async () => {
      const res = await httpJson('/artifacts', {
        method: 'POST',
        headers: withAuthHeaders(token, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({ query }),
      })
      return res.error
        ? { error: res.error }
        : { data: { artifacts: extractArtifacts(res.data, fallbackType) } }
    },
    async () => {
      const res = await httpJson(`/artifact/byName/${encodeURIComponent(query)}`, {
        method: 'GET',
        headers: withAuthHeaders(token),
      })
      return res.error
        ? { error: res.error }
        : { data: { artifacts: extractArtifacts(res.data, fallbackType) } }
    },
    async () => {
      const res = await httpJson('/artifact/byRegEx', {
        method: 'POST',
        headers: withAuthHeaders(token, { 'Content-Type': 'application/json' }),
        body: JSON.stringify({ query }),
      })
      return res.error
        ? { error: res.error }
        : { data: { artifacts: extractArtifacts(res.data, fallbackType) } }
    },
  ]

  const errors: string[] = []

  for (const [index, run] of strategies.entries()) {
    const res = await run()
    if (res.error) {
      errors.push(res.error)
      continue
    }
    const artifacts = res.data?.artifacts ?? []
    if (artifacts.length) return { data: { artifacts, source: ['search', 'name', 'regex'][index] } }
  }

  if (errors.length) {
    return { error: errors.join(' | ') }
  }

  return { data: { artifacts: [], source: 'none' } }
}

export const fetchRatings = async (id: string, token?: string) =>
  httpJson(`/artifact/model/${encodeURIComponent(id)}/rate`, {
    method: 'GET',
    headers: withAuthHeaders(token),
  })

export const fetchCost = async (artifactType: string, id: string, token?: string) =>
  httpJson(`/artifact/${encodeURIComponent(artifactType)}/${encodeURIComponent(id)}/cost`, {
    method: 'GET',
    headers: withAuthHeaders(token),
  })

export const fetchLineage = async (id: string, token?: string) =>
  httpJson(`/artifact/model/${encodeURIComponent(id)}/lineage`, {
    method: 'GET',
    headers: withAuthHeaders(token),
  })

export const deleteArtifact = async (artifactType: string, id: string, token?: string) =>
  httpJson(`/artifacts/${encodeURIComponent(artifactType)}/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    headers: withAuthHeaders(token),
  })

export const downloadArtifact = async (artifactType: string, id: string, token?: string) =>
  httpJson<{ url?: string; downloadUrl?: string }>(
    `/artifacts/${encodeURIComponent(artifactType)}/${encodeURIComponent(id)}`,
    { method: 'GET', headers: withAuthHeaders(token) },
  )

export const checkLicense = async (id: string, token?: string) =>
  httpJson(`/artifact/model/${encodeURIComponent(id)}/license-check`, {
    method: 'POST',
    headers: withAuthHeaders(token),
  })

export const fetchAudits = async (artifactType: string, id: string, token?: string) =>
  httpJson(`/artifact/${encodeURIComponent(artifactType)}/${encodeURIComponent(id)}/audit`, {
    method: 'GET',
    headers: withAuthHeaders(token),
  })
