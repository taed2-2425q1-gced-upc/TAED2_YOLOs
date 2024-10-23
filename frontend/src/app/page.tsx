"use client"
import {useState} from "react"
import PageHeader from "@/components/layout/PageHeader"
import PageSubtitle from "@/components/text/PageSubtitle"
import PageTitle from "@/components/text/PageTitle"
import {Image} from "@/infraestructure/types"
import {
  PredictionData,
  predictService,
} from "@/infraestructure/predictionsService"
import {ImageUpload} from "@/components/image/ImageUpload"
import toast from "react-hot-toast"
import {PredictionCard} from "@/components/predictions/PredictionCard"
import {PredictionStatsContainer} from "@/components/predictions/PredictionStats"

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionData | null>(null)
  const [original, setOriginal] = useState<Image | null>(null)

  const handlePrediction = async (img: Image, hasStats: boolean) => {
    setPrediction(null)
    setOriginal(img)

    try {
      toast.loading("Predicting...")

      console.log("predicting with stats?", hasStats)

      const prediction = hasStats
        ? await predictService.getPredictionWithStats(img)
        : await predictService.getPrediction(img)

      if (!prediction.mask) {
        throw new Error("No mask found")
      }

      console.log(prediction)

      setPrediction(prediction)
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
      <PredictionCard prediction={prediction?.mask!} original={original!} />
      <PredictionStatsContainer stats={prediction?.stats!} />
    </main>
  )
}
