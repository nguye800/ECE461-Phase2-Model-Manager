'use client'

import * as React from 'react'
import { useEffect, useState } from 'react'
import { ShieldCheck, ShieldAlert, Loader2 } from 'lucide-react'

import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card'
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert'
import { SearchTab } from '@/components/registry/search-tab'
import { UploadTab } from '@/components/registry/upload-tab'
import { authenticate } from '@/lib/registry-api'

type AuthStatus = 'loading' | 'ok' | 'error'

export default function Page() {
  const [token, setToken] = useState<string | undefined>()
  const [authStatus, setAuthStatus] = useState<AuthStatus>('loading')
  const [authError, setAuthError] = useState<string | undefined>()

  useEffect(() => {
    let cancelled = false

    const run = async () => {
      setAuthStatus('loading')
      const res = await authenticate()

      if (cancelled) return

      if (res.error) {
        setAuthStatus('error')
        setAuthError(res.error)
        setToken(undefined)
      } else {
        setAuthStatus('ok')
        setToken(res.data?.token)
        setAuthError(undefined)
      }
    }

    run()

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <main className="min-h-screen bg-background">
      <div className="container mx-auto max-w-5xl py-8 space-y-6">
        {/* Header */}
        <Card>
          <CardHeader>
            <CardTitle className="text-3xl">Trustworthy Model Registry</CardTitle>
            <CardDescription>
              Web UI for searching, inspecting, and uploading artifacts using your Phase 2 API.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {/* Auth status */}
            {authStatus === 'loading' && (
              <Alert>
                <AlertTitle className="flex items-center gap-2">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Requesting token…
                </AlertTitle>
                <AlertDescription>
                  Contacting the <code>/authenticate</code> endpoint to get an authorization token.
                </AlertDescription>
              </Alert>
            )}

            {authStatus === 'ok' && (
              <Alert>
                <AlertTitle className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4 text-emerald-600" />
                  Authenticated
                </AlertTitle>
                <AlertDescription className="text-xs text-muted-foreground">
                  Token acquired from <code>/authenticate</code>. All requests from this page will
                  include it.
                  {token && (
                    <div className="mt-1 font-mono break-all">
                      token: {token.slice(0, 32)}…
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}

            {authStatus === 'error' && (
              <Alert variant="destructive">
                <AlertTitle className="flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4" />
                  Authentication failed
                </AlertTitle>
                <AlertDescription className="text-xs">
                  {authError}
                  <br />
                  You can still try using the registry without a token, but requests may be rejected
                  by the backend if authentication is required.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        {/* Main tabs */}
        <Tabs defaultValue="search" className="w-full">
          <TabsList className="mb-4">
            <TabsTrigger value="search">Search</TabsTrigger>
            <TabsTrigger value="upload">Upload</TabsTrigger>
          </TabsList>

          <TabsContent value="search">
            <SearchTab token={token} />
          </TabsContent>

          <TabsContent value="upload">
            <UploadTab token={token} />
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
