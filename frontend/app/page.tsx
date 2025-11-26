"use client"

import { ShieldCheck } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { UploadTab } from "@/components/registry/upload-tab"
import { SearchTab } from "@/components/registry/search-tab"

export default function Home() {
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

      <Tabs defaultValue="upload" className="space-y-4">
        <TabsList>
          <TabsTrigger value="upload">Upload Artifact</TabsTrigger>
          <TabsTrigger value="search">Search Registry</TabsTrigger>
        </TabsList>
        <TabsContent value="upload">
          <UploadTab />
        </TabsContent>
        <TabsContent value="search">
          <SearchTab />
        </TabsContent>
      </Tabs>
    </main>
  )
}
