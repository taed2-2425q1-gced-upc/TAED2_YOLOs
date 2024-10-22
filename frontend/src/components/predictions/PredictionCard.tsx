import {Image} from "@/infraestructure/types"

export interface PredictionCardProps {
  prediction: Image
  original: Image
}

export const PredictionCard: React.FC<PredictionCardProps> = ({
  prediction,
  original,
}) => {
  if (!prediction || !prediction.url) {
    return null
  }
  return (
    <div className="flex flex-col items-center justify-center w-100 h-full my-8 gap-3">
      <h2>Prediction</h2>
      <div className="relative">
        <img className=" w-full h-full" src={original.url} />
        <img
          className="absolute top-0 left-0 w-full h-full mix-blend-lighten opacity-50"
          src={prediction.url}
          alt="Prediction"
        />
      </div>
    </div>
  )
}
