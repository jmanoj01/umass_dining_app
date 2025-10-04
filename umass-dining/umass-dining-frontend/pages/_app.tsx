import '../styles/globals.css';
import type { AppProps } from 'next/app';
import { useEffect } from 'react';

export default function MyApp({ Component, pageProps }: AppProps) {
  // Set Inter font globally
  useEffect(() => {
    document.body.style.fontFamily = 'Inter, ui-sans-serif, system-ui';
  }, []);
  return <Component {...pageProps} />;
}
