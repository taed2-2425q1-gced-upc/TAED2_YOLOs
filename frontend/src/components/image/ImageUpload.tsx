"use client"
import {useEffect, useState} from "react"
import PrimaryButton from "../buttons/PrimaryButton"
import ImageInput, {ImageName} from "./ImageInput"
import {Image, toImage} from "@/infraestructure/types"
import {StatsCheckbox} from "../stats/GetStatsSelect"

interface ImageUploadProps {
  onSubmit: (img: Image, hasStats: boolean) => void
}

export const ImageUpload: React.FC<ImageUploadProps> = ({onSubmit}) => {
  const [isImageUploaded, setIsImageUploaded] = useState(false)
  const [image, setImage] = useState<File | null>(null)
  const [getStats, setGetStats] = useState(false)

  const handleImageUpload = (uploaded: boolean, image: File | null) => {
    setImage(image)
    setIsImageUploaded(uploaded)
  }

  const handleGetStats = (checked: boolean) => {
    setGetStats(checked)
  }

  return (
    <div>
      <ImageInput onImageUpload={handleImageUpload} />
      <StatsCheckbox onChange={handleGetStats} />
      <PrimaryButton
        onClick={() => {
          onSubmit(toImage(image!), getStats)
          setImage(null)
        }}
        disabled={!isImageUploaded}
      >
        Submit
      </PrimaryButton>
      <ImageName image={image} />
    </div>
  )
}
