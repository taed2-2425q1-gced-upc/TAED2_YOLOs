"use client" // Ensures the component runs only on the client

import React from "react"

interface PrimaryButtonProps {
  children: React.ReactNode
  onClick?: () => void // Optional onClick handler
  disabled: boolean
}

const PrimaryButton: React.FC<PrimaryButtonProps> = ({
  children,
  onClick,
  disabled,
}) => {
  return (
    <button
      className="disabled:opacity-20 disabled:hover:bg-blue-500 bg-blue-500 hover:bg-blue-700 text-white py-2 px-4 w-full font-light rounded-lg"
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  )
}

export default PrimaryButton
