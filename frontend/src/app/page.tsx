"use client"
import {useState} from "react"
import PageHeader from "@/components/layout/PageHeader"
import PageSubtitle from "@/components/text/PageSubtitle"
import PageTitle from "@/components/text/PageTitle"
import {Image} from "@/infraestructure/types"
import {predictService} from "@/infraestructure/predictionsService"
import {ImageUpload} from "@/components/image/ImageUpload"
import toast from "react-hot-toast"
import {PredictionCard} from "@/components/predictions/PredictionCard"

export default function Home() {
  const [prediction, setPrediction] = useState<Image | null>(null)

  const handlePrediction = async (img: Image) => {
    toast.promise(
      predictService.getPrediction(img).then((prediction) => {
        if (!prediction.mask) {
          throw new Error("No mask found")
        }
        setPrediction(prediction.mask)
      }),
      {
        loading: "Predicting...",
        success: <b>Prediction complete!</b>,
        error: <b>Prediction failed!</b>,
      }
    )
  }

  return (
    <main className="font-mono">
      <PageHeader>
        <PageTitle>Person Segmentation</PageTitle>
        <PageSubtitle>with YOLOv8-Seg</PageSubtitle>
      </PageHeader>
      <ImageUpload onSubmit={handlePrediction} />
      <PredictionCard prediction={prediction!} />
    </main>
  )
}
