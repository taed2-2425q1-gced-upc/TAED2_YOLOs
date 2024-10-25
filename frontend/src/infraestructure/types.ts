export interface Image {
  name: string
  url?: string
  file?: File
}

export function toImage(file: File): Image {
  return {
    name: file.name,
    file: file,
    url: URL.createObjectURL(file),
  }
}

export interface PredictionStats {
  emissions: number
  duration: number
  cpu_power: number
  gpu_power: number
  ram_power: number
  energy_consumed: number
}
