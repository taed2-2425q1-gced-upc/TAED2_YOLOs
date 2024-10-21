import {Image} from "@/infraestructure/types"

export interface PredictionCardProps {
  prediction: Image
}

export const PredictionCard: React.FC<PredictionCardProps> = ({prediction}) => {
  if (!prediction || !prediction.url) {
    return null
  }
  return (
    <div className="flex flex-col items-center justify-center w-100 h-full my-8 gap-3">
      <h2>Prediction</h2>
      <img src={prediction.url} alt="Prediction" />
    </div>
  )
}
