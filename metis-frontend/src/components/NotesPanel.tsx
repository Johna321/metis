import { useState } from "react";
import { IoClose, IoCheckmark } from "react-icons/io5";
import type { BBoxSelection } from "./PdfViewer";

export interface Note {
  id: string;
  docId: string;
  bbox: BBoxSelection;
  query: string;
  response: string;
  createdAt: number;
}

interface NotesPanelProps {
  note: Note | null;
  onClose: () => void;
  onHighlightBBox: () => void;
}

export function NotesPanel({ note, onClose, onHighlightBBox }: NotesPanelProps) {
  const [isHighlighting, setIsHighlighting] = useState(false);

  if (!note) return null;

  const handleHighlight = () => {
    setIsHighlighting(true);
    onHighlightBBox();
    setTimeout(() => setIsHighlighting(false), 200);
  };

  const createdDate = new Date(note.createdAt).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="notes-panel">
      <div className="notes-panel-header">
        <h3 className="notes-panel-title">Note</h3>
        <button
          className="notes-panel-close"
          onClick={onClose}
          aria-label="Close note"
        >
          <IoClose />
        </button>
      </div>

      <div className="notes-panel-content">
        <div className="notes-panel-section">
          <div className="notes-panel-label">Query</div>
          <div className="notes-panel-query">{note.query}</div>
        </div>

        <div className="notes-panel-section">
          <div className="notes-panel-label">Response</div>
          <div className="notes-panel-response">{note.response}</div>
        </div>

        <div className="notes-panel-meta">
          <span className="notes-panel-timestamp">{createdDate}</span>
        </div>
      </div>

      <div className="notes-panel-footer">
        <button
          className={`notes-panel-highlight-btn ${isHighlighting ? "active" : ""}`}
          onClick={handleHighlight}
        >
          {isHighlighting ? <IoCheckmark /> : "✓"} Show Selection
        </button>
      </div>
    </div>
  );
}
