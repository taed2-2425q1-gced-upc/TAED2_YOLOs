export default function PageHeader({
  children,
}: Readonly<{children: React.ReactNode}>) {
  return (
    <div className="flex flex-col items-center justify-center w-full h-full my-8">
      {children}
    </div>
  )
}
