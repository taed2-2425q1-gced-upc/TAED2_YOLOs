"use client"
import {useState} from "react"
import PrimaryButton from "../buttons/PrimaryButton"
import ImageInput, {ImageName} from "./ImageInput"

export default function ImageUpload() {
  const [isImageUploaded, setIsImageUploaded] = useState(false)
  const [image, setImage] = useState<File | null>(null)

  const handleImageUpload = (uploaded: boolean, image: File | null) => {
    setImage(image)
    setIsImageUploaded(uploaded)
  }
  return (
    <div>
      <ImageInput onImageUpload={handleImageUpload} />
      <PrimaryButton disabled={!isImageUploaded}>Submit</PrimaryButton>
      <ImageName image={image} />
    </div>
  )
}
