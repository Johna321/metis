import { Quantum } from 'ldrs/react'
import { Grid } from 'ldrs/react'
import 'ldrs/react/Quantum.css'
import 'ldrs/react/Grid.css'

interface ToolbarProps {
  onOpenPdf: () => void;
  onBBoxToggle: () => void;
  bboxMode: boolean;
  docId: string | null;
  status: string;
  numPages: number;
  isChatMinimized: boolean;
  onToggleChatMinimize: () => void;
}

export function Toolbar({ onOpenPdf, onBBoxToggle, bboxMode, docId, status, numPages, isChatMinimized, onToggleChatMinimize }: ToolbarProps) {
  return (
    <header className="toolbar">
      <button onClick={onOpenPdf}>Open PDF</button>

      <button
        className={`bbox-btn${bboxMode ? " active" : ""}`}
        onClick={onBBoxToggle}
        disabled={!docId}
      >
        BBox
      </button>

      <div className="spacer" />
      <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        {status}
        {status === "Ingesting..." && <Quantum size="25" speed="1.75" color="black" />}
        {status === "Vectorizing..." && <Grid size="25" speed="1.5" color="black" />}
        {status === "Loaded (cached)" && " \u2713"}
      </span>
      <span style={{ marginLeft: 12 }}>{numPages ? `${numPages} pages` : ""}</span>

      <button
        className="chat-minimize-btn"
        onClick={onToggleChatMinimize}
        title={isChatMinimized ? "Expand chat" : "Minimize chat"}
      >
        {isChatMinimized ? "◀" : "▶"}
      </button>
    </header>
  );
}
