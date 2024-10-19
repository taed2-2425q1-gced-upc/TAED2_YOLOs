export interface Image {
  name: string
  url?: string
  file?: File
}

export function toImage(file: File): Image {
  return {
    name: file.name,
    file: file,
  }
}
