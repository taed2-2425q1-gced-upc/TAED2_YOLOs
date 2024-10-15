export default function PageSubtitle({
  children,
}: Readonly<{children: React.ReactNode}>) {
  return (
    <h2 className="pt-2 font-light text-center text-gray-400">{children}</h2>
  )
}
