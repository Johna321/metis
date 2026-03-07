interface ToolbarProps {
  onOpenPdf: () => void;
  onBBoxToggle: () => void;
  bboxMode: boolean;
  docId: string | null;
  status: string;
  numPages: number;
}

export function Toolbar({ onOpenPdf, onBBoxToggle, bboxMode, docId, status, numPages }: ToolbarProps) {
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
      <span>{status}</span>
      <span style={{ marginLeft: 12 }}>{numPages ? `${numPages} pages` : ""}</span>
    </header>
  );
}
