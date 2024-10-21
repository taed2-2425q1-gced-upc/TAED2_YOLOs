import type {Metadata} from "next"
import {GeistMono} from "geist/font/mono"
import "./globals.css"
import {Toaster} from "react-hot-toast"

const geistMono = GeistMono.variable

export const metadata: Metadata = {
  title: "Person segmentation",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={geistMono}>
      <body>
        <main className="flex flex-col items-center py-10 px-8 box-border min-h-100vh max-w-screen-sm m-auto">
          {children}
        </main>
      </body>
      <Toaster />
    </html>
  )
}
