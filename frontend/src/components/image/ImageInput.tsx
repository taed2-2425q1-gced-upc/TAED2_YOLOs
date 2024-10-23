"use client"
import React, {useState, useRef} from "react"
import Image from "next/image"

interface ImageInputProps {
  containerClassName?: string
  imageClassName?: string
  onImageUpload: (uploaded: boolean, image: File | null) => void // Add this prop
}

const ImageInput: React.FC<ImageInputProps> = ({
  containerClassName = "",
  imageClassName = "",
  onImageUpload,
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      onImageUpload(true, file) // Notify the parent that an image was uploaded
    } else {
      onImageUpload(false, null)
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div
      className={`relative ${containerClassName} my-2`}
      onClick={handleClick}
    >
      <input
        type="file"
        accept="image/*"
        onChange={handleImageUpload}
        className="hidden"
        ref={fileInputRef}
      />
      <div className="flex items-center justify-center w-full h-full rounded-xl border-spacing-3 border-dashed border py-7 px-9 text-gray-500 text-sm cursor-pointer">
        Click to upload an image
      </div>
    </div>
  )
}

export default ImageInput

interface AddedImageProps {
  image: File | null // Change type to File
}

export const ImageName: React.FC<AddedImageProps> = ({image}) => {
  if (!image) return null // Return null if no image

  return (
    <div className="mt-3">
      <p className="text-gray-400 text-sm text-center">{image.name}</p>
    </div>
  )
}
