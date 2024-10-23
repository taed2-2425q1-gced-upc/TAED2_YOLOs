import axios from "axios"
import {Image, PredictionStats} from "./types"

const BASE_URL = "http://127.0.0.1:8000"

const headers = {
  "Content-Type": "multipart/form-data",
  Authorization: "Bearer YOLOsImageSegmentation",
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
      const response = await axios.post(BASE_URL + "/predict", formData, {
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
      const response = await axios.post(BASE_URL + "/predict", formData, {
        headers,
      })

      const prediction: PredictionData = {
        originalImage: img,
        mask: {
          name: response.data.filename,
          url: BASE_URL + "/static/" + response.data.filename,
        },
        stats: {
          mIoU: 45.0,
          f1: 23,
          precision: 0.86,
          recall: 0.93,
        },
      }
      return prediction
    } catch (err) {
      console.log(err)
      throw err
    }
  }
}

export const predictService = new Prediction()
