"use client"

import { useState } from "react"
import { KeyRound, ShieldCheck, ShieldQuestion } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { UploadTab } from "@/components/registry/upload-tab"
import { SearchTab } from "@/components/registry/search-tab"
import { authenticate } from "@/lib/registry-api"

export default function Home() {
  const [token, setToken] = useState("")
  const [authStatus, setAuthStatus] = useState<{ message: string; tone: "idle" | "error" | "success" }>(
    { message: "", tone: "idle" },
  )

  const requestToken = async () => {
    setAuthStatus({ message: "Requesting access token...", tone: "idle" })
    const res = await authenticate()
    if (res.error) {
      setAuthStatus({ message: res.error, tone: "error" })
      return
    }
    const newToken = res.data?.token
    if (newToken) {
      setToken(newToken)
      setAuthStatus({ message: "Token acquired and applied to subsequent requests.", tone: "success" })
    } else {
      setAuthStatus({
        message: "Authenticate endpoint responded without a token. You can paste an existing token manually.",
        tone: "error",
      })
    }
  }

  return (
    <main className="mx-auto flex max-w-6xl flex-col gap-6 p-6">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-3xl font-bold">Model Registry Dashboard</h1>
            <p className="text-muted-foreground">
              Upload artifacts or search the registry using the Hugging Face Model Manager endpoints.
            </p>
          </div>
          <Badge variant="secondary" className="gap-1">
            <ShieldCheck className="h-4 w-4" />
            Client-side UI
          </Badge>
        </div>
      </div>

      <Card className="shadow-sm">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <ShieldQuestion className="h-5 w-5" />
            Authentication (optional)
          </CardTitle>
          <CardDescription>
            Use PUT /authenticate to generate a token, or paste an existing one. The token (if provided) is sent as a
            Bearer header for all actions below.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid gap-3 md:grid-cols-[1fr,fit-content(100px)]">
            <div className="space-y-2">
              <Label htmlFor="token">Access token</Label>
              <Input
                id="token"
                placeholder="Paste token or request one"
                value={token}
                onChange={(e) => setToken(e.target.value)}
                autoComplete="off"
              />
            </div>
            <div className="flex items-end">
              <Button type="button" className="w-full gap-2" onClick={requestToken}>
                <KeyRound className="h-4 w-4" />
                Get token
              </Button>
            </div>
          </div>
          {authStatus.message && (
            <Alert variant={authStatus.tone === "error" ? "destructive" : "default"}>
              <AlertTitle>{authStatus.tone === "error" ? "Authentication failed" : "Authentication"}</AlertTitle>
              <AlertDescription>{authStatus.message}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      <Tabs defaultValue="upload" className="space-y-4">
        <TabsList>
          <TabsTrigger value="upload">Upload Artifact</TabsTrigger>
          <TabsTrigger value="search">Search Registry</TabsTrigger>
        </TabsList>
        <TabsContent value="upload">
          <UploadTab token={token} />
        </TabsContent>
        <TabsContent value="search">
          <SearchTab token={token} />
        </TabsContent>
      </Tabs>
    </main>
  )
}
