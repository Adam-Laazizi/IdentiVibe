import React, { useState, useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Home } from './routes/Home';
import { Sources } from './routes/Sources';
import { Results } from './routes/Results';
import type { ResolveSourcesResponse } from './types/sources';
import './index.css';

type Route = 'home' | 'sources' | 'results';

function useRouter() {
  const [route, setRoute] = useState<Route>('home');
  const [state, setState] = useState<ResolveSourcesResponse | null>(null);

  useEffect(() => {
  const handleRoute = () => {
    const hash = window.location.hash.slice(1);
    if (hash === '/sources') setRoute('sources');
    else if (hash === '/results') setRoute('results');
    else setRoute('home');
  };

  handleRoute();
  window.addEventListener('hashchange', handleRoute);
  return () => window.removeEventListener('hashchange', handleRoute);
}, []);


  const navigate = (path: string, newState?: ResolveSourcesResponse) => {
    window.location.hash = path;
    if (newState !== undefined) {
      setState(newState);
    }
  };

  return { route, state, navigate };
}

function App() {
  const { route, state, navigate } = useRouter();

  if (route === 'sources') {
    return <Sources initialState={state} navigate={navigate} />;
  }

  if (route === 'results') {
    return <Results initialState={state} navigate={navigate} />;
  }

  return <Home navigate={navigate} />;
}

const root = document.getElementById('root');
if (root) {
  createRoot(root).render(<App />);
}