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
  const [original, setOriginal] = useState<Image | null>(null)

  const handlePrediction = async (img: Image) => {
    setPrediction(null)
    setOriginal(img)

    try {
      toast.loading("Predicting...")

      const prediction = await predictService.getPrediction(img)

      if (!prediction.mask) {
        throw new Error("No mask found")
      }

      setPrediction(prediction.mask)
      toast.dismiss() // Dismiss the loading toast
      toast.success(<b>Prediction complete!</b>)
    } catch (error: any) {
      toast.dismiss() // Dismiss the loading toast

      if (error.response && error.response.status === 400) {
        toast.error("No masks found")
      } else {
        toast.error("Prediction failed!")
      }
    }
  }

  return (
    <main className="font-mono">
      <PageHeader>
        <PageTitle>Person Segmentation</PageTitle>
        <PageSubtitle>with YOLOv8-Seg</PageSubtitle>
      </PageHeader>
      <ImageUpload onSubmit={handlePrediction} />
      <PredictionCard prediction={prediction!} original={original!} />
    </main>
  )
}
