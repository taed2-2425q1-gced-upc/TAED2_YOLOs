"use client"
import {useState} from "react"
import PrimaryButton from "../buttons/PrimaryButton"
import ImageInput, {ImageName} from "./ImageInput"
import {Image, toImage} from "@/infraestructure/types"

interface ImageUploadProps {
  onSubmit: (img: Image) => void
}

export const ImageUpload: React.FC<ImageUploadProps> = ({onSubmit}) => {
  const [isImageUploaded, setIsImageUploaded] = useState(false)
  const [image, setImage] = useState<File | null>(null)

  const handleImageUpload = (uploaded: boolean, image: File | null) => {
    setImage(image)
    setIsImageUploaded(uploaded)
  }
  return (
    <div>
      <ImageInput onImageUpload={handleImageUpload} />
      <PrimaryButton
        onClick={() => {
          onSubmit(toImage(image!))
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
