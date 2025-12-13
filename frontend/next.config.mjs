/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    unoptimized: true,
  },
  reactStrictMode: false,
  swcMinify: true,
  experimental: {
    optimizePackageImports: ["@radix-ui/react-*"],
  },
}

export default nextConfig
