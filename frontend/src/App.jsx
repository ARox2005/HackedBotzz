import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import './App.css';

function App() {
  const [selectedSources, setSelectedSources] = useState([]);
  const [useAllDocs, setUseAllDocs] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="app">
      {/* Mobile menu button */}
      <button
        className="mobile-menu-btn"
        onClick={() => setSidebarOpen(!sidebarOpen)}
        aria-label="Toggle menu"
      >
        {sidebarOpen ? '✕' : '☰'}
      </button>

      {/* Overlay for mobile */}
      <div
        className={`sidebar-overlay ${sidebarOpen ? 'visible' : ''}`}
        onClick={() => setSidebarOpen(false)}
      />

      <Sidebar
        selectedSources={selectedSources}
        setSelectedSources={setSelectedSources}
        useAllDocs={useAllDocs}
        setUseAllDocs={setUseAllDocs}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />
      <ChatWindow
        selectedSources={selectedSources}
        useAllDocs={useAllDocs}
      />
    </div>
  );
}

export default App;
