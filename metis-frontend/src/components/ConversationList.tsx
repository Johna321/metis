import { useEffect, useRef, useState } from "react";
import type { ConversationMeta } from "../backend/http";

interface ConversationListProps {
  conversations: ConversationMeta[];
  activeConvId: string | null;
  isCollapsed: boolean;
  onSelect: (convId: string) => void;
  onCreate: () => void;
  onRename: (convId: string, title: string) => void;
  onDelete: (convId: string) => void;
  onPin: (convId: string, pinned: boolean) => void;
  onToggleCollapse: () => void;
}

export function ConversationList({
  conversations,
  activeConvId,
  isCollapsed,
  onSelect,
  onCreate,
  onRename,
  onDelete,
  onPin,
  onToggleCollapse,
}: ConversationListProps) {
  const [contextMenu, setContextMenu] = useState<{ convId: string; x: number; y: number } | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  useEffect(() => {
    function handleClick() { setContextMenu(null); }
    if (contextMenu) {
      document.addEventListener("click", handleClick);
      return () => document.removeEventListener("click", handleClick);
    }
  }, [contextMenu]);

  function handleContextMenu(e: React.MouseEvent, convId: string) {
    e.preventDefault();
    setContextMenu({ convId, x: e.clientX, y: e.clientY });
  }

  function startRename(convId: string, currentTitle: string) {
    setRenamingId(convId);
    setRenameValue(currentTitle);
    setContextMenu(null);
  }

  function commitRename() {
    if (renamingId && renameValue.trim()) {
      onRename(renamingId, renameValue.trim());
    }
    setRenamingId(null);
  }

  function handleRenameKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter") commitRename();
    if (e.key === "Escape") setRenamingId(null);
  }

  const pinned = conversations.filter(c => c.pinned);
  const unpinned = conversations.filter(c => !c.pinned);

  if (isCollapsed) {
    return (
      <div className="conv-list conv-list--collapsed">
        <button className="conv-collapse-btn" onClick={onToggleCollapse} title="Expand conversations">&#9654;</button>
        <button className="conv-new-btn-icon" onClick={onCreate} title="New conversation">+</button>
      </div>
    );
  }

  return (
    <div className="conv-list">
      <div className="conv-list-header">
        <span className="conv-list-label">Conversations</span>
        <div className="conv-list-actions">
          <button className="conv-new-btn-icon" onClick={onCreate} title="New conversation">+</button>
          <button className="conv-collapse-btn" onClick={onToggleCollapse} title="Collapse list">&#9664;</button>
        </div>
      </div>

      <div className="conv-list-items">
        {pinned.length > 0 && (
          <>
            <div className="conv-section-label">Pinned</div>
            {pinned.map(c => (
              <ConvItem
                key={c.id}
                conv={c}
                isActive={c.id === activeConvId}
                isRenaming={c.id === renamingId}
                renameValue={renameValue}
                renameInputRef={c.id === renamingId ? renameInputRef : undefined}
                onSelect={() => onSelect(c.id)}
                onContextMenu={(e) => handleContextMenu(e, c.id)}
                onRenameChange={setRenameValue}
                onRenameKeyDown={handleRenameKeyDown}
                onRenameBlur={commitRename}
              />
            ))}
          </>
        )}

        {unpinned.length > 0 && (
          <>
            {pinned.length > 0 && <div className="conv-section-label">Recent</div>}
            {unpinned.map(c => (
              <ConvItem
                key={c.id}
                conv={c}
                isActive={c.id === activeConvId}
                isRenaming={c.id === renamingId}
                renameValue={renameValue}
                renameInputRef={c.id === renamingId ? renameInputRef : undefined}
                onSelect={() => onSelect(c.id)}
                onContextMenu={(e) => handleContextMenu(e, c.id)}
                onRenameChange={setRenameValue}
                onRenameKeyDown={handleRenameKeyDown}
                onRenameBlur={commitRename}
              />
            ))}
          </>
        )}

        {conversations.length === 0 && (
          <div className="conv-empty">No conversations yet</div>
        )}
      </div>

      {contextMenu && (
        <div className="conv-context-menu" style={{ position: "fixed", left: contextMenu.x, top: contextMenu.y, zIndex: 1000 }}>
          {(() => {
            const conv = conversations.find(c => c.id === contextMenu.convId);
            return (
              <>
                <button className="conv-context-item" onClick={() => { onPin(contextMenu.convId, !conv?.pinned); setContextMenu(null); }}>
                  {conv?.pinned ? "Unpin" : "Pin"} conversation
                </button>
                <button className="conv-context-item" onClick={() => startRename(contextMenu.convId, conv?.title ?? "")}>
                  Rename
                </button>
                <div className="conv-context-divider" />
                <button className="conv-context-item conv-context-item--danger" onClick={() => { onDelete(contextMenu.convId); setContextMenu(null); }}>
                  Delete
                </button>
              </>
            );
          })()}
        </div>
      )}
    </div>
  );
}

function ConvItem({
  conv,
  isActive,
  isRenaming,
  renameValue,
  renameInputRef,
  onSelect,
  onContextMenu,
  onRenameChange,
  onRenameKeyDown,
  onRenameBlur,
}: {
  conv: ConversationMeta;
  isActive: boolean;
  isRenaming: boolean;
  renameValue: string;
  renameInputRef?: React.RefObject<HTMLInputElement | null>;
  onSelect: () => void;
  onContextMenu: (e: React.MouseEvent) => void;
  onRenameChange: (v: string) => void;
  onRenameKeyDown: (e: React.KeyboardEvent) => void;
  onRenameBlur: () => void;
}) {
  const relativeTime = formatRelativeTime(conv.updated_at);

  return (
    <div
      className={`conv-item ${isActive ? "conv-item--active" : ""}`}
      onClick={onSelect}
      onContextMenu={onContextMenu}
    >
      <span className="conv-pin-icon">{conv.pinned ? "\u2605" : ""}</span>
      <div className="conv-item-content">
        {isRenaming ? (
          <input
            ref={renameInputRef}
            className="conv-rename-input"
            value={renameValue}
            onChange={e => onRenameChange(e.target.value)}
            onKeyDown={onRenameKeyDown}
            onBlur={onRenameBlur}
            onClick={e => e.stopPropagation()}
          />
        ) : (
          <div className="conv-item-title">{conv.title}</div>
        )}
        <div className="conv-item-meta">
          {conv.message_count} msg{conv.message_count !== 1 ? "s" : ""}
          {relativeTime && <> &middot; {relativeTime}</>}
        </div>
      </div>
    </div>
  );
}

function formatRelativeTime(isoString: string): string {
  const date = new Date(isoString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return "now";
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay}d ago`;
  return date.toLocaleDateString();
}
