import axios from "axios"
import {Image, PredictionStats} from "./types"

const BASE_URL = process.env.NEXT_PUBLIC_BASE_URL

const headers = {
  "Content-Type": "multipart/form-data",
  Authorization: `Bearer ${process.env.NEXT_PUBLIC_AUTH_TOKEN}`,
}

export interface PredictionData {
  originalImage?: Image
  mask?: Image
  stats?: PredictionStats
}

interface IPrediction {
  originalImage?: Image
  mask?: Image

  getPrediction(img: Image): Promise<PredictionData>
  getPredictionWithStats(img: Image): Promise<PredictionData>
}

class Prediction implements IPrediction {
  originalImage?: Image
  mask?: Image

  constructor() {}

  async getPrediction(img: Image): Promise<Prediction> {
    const formData = new FormData()

    try {
      if (!img.file) {
        throw new Error("No file selected")
      }

      formData.append("file", img.file)
      const response = await axios.post(BASE_URL + "/predict/image", formData, {
        headers,
      })

      this.mask = {
        name: response.data.filename,
        url: BASE_URL + "/static/" + response.data.filename,
      }
      return this
    } catch (err) {
      console.log(err)
      throw err
    }
  }

  async getPredictionWithStats(img: Image): Promise<PredictionData> {
    const formData = new FormData()
    try {
      if (!img.file) {
        throw new Error("No file selected")
      }

      formData.append("file", img.file)
      const response = await axios.post(
        BASE_URL + "/predict/image/emissions",
        formData,
        {
          headers,
        }
      )

      const prediction: PredictionData = {
        originalImage: img,
        mask: {
          name: response.data.prediction.filename,
          url: BASE_URL + "/static/" + response.data.prediction.filename,
        },
        stats: response.data.energy_stats,
      }
      return prediction
    } catch (err) {
      console.log(err)
      throw err
    }
  }
}

export const predictService = new Prediction()
