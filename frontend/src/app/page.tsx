import {useState} from "react"
import PageHeader from "@/components/layout/PageHeader"
import PageSubtitle from "@/components/text/PageSubtitle"
import PageTitle from "@/components/text/PageTitle"
import ImageUpload from "@/components/image/ImageUpload"

export default function Home() {
  return (
    <div className="font-mono">
      <PageHeader>
        <PageTitle>Person Segmentation</PageTitle>
        <PageSubtitle>with YOLOv8-Seg</PageSubtitle>
      </PageHeader>
      <ImageUpload />
    </div>
  )
}
