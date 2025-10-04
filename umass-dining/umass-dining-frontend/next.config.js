/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  images: {
    domains: ['localhost'],
  },
  webpack(config) {
    return config;
  },
}

module.exports = nextConfig;
