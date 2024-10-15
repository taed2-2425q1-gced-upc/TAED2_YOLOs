export default function PageTitle({
  children,
}: Readonly<{children: React.ReactNode}>) {
  return <h1 className="text-4xl font-bold text-center">{children}</h1>
}
