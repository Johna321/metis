import { useState, useEffect } from "react";
import { readSettings, saveSettings, type ApiSettings } from "../backend/http";

interface SettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

const PROVIDERS = [
  { value: "anthropic", label: "Anthropic" },
  { value: "openai", label: "OpenAI" },
  { value: "openrouter", label: "OpenRouter" },
];

const MODELS: Record<string, string[]> = {
  anthropic: [
    "claude-opus-4-6",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
  ],
  openai: ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
  openrouter: ["meta-llama/llama-2-70b-chat", "mistralai/mistral-7b-instruct"],
};

export function Settings({ isOpen, onClose }: SettingsProps) {
  const [settings, setSettings] = useState<ApiSettings>({
    provider: "anthropic",
    model: "claude-sonnet-4-20250514",
  });
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (isOpen) {
      loadSettings();
    }
  }, [isOpen]);

  async function loadSettings() {
    try {
      setLoading(true);
      const loaded = await readSettings();
      setSettings(loaded);
      setError(null);
    } catch (err) {
      setError(`Failed to load settings: ${err}`);
    } finally {
      setLoading(false);
    }
  }

  async function handleSave() {
    try {
      setSaving(true);
      setError(null);
      await saveSettings(settings);
      setSuccess(true);
      setTimeout(() => setSuccess(false), 2000);
    } catch (err) {
      setError(`Failed to save settings: ${err}`);
    } finally {
      setSaving(false);
    }
  }

  const availableModels =
    MODELS[settings.provider] || MODELS.anthropic;

  if (!isOpen) return null;

  return (
    <div className="settings-modal-overlay" onClick={onClose}>
      <div className="settings-modal" onClick={(e) => e.stopPropagation()}>
        <div className="settings-header">
          <h2>Settings</h2>
          <button className="close-btn" onClick={onClose}>
            x
          </button>
        </div>

        <div className="settings-content">
          {loading ? (
            <p>Loading settings...</p>
          ) : (
            <>
              <div className="settings-section">
                <label htmlFor="provider">LLM Provider</label>
                <select
                  id="provider"
                  value={settings.provider}
                  onChange={(e) => {
                    const newProvider = e.target.value;
                    setSettings({
                      ...settings,
                      provider: newProvider,
                      model:
                        MODELS[newProvider]?.[0] || settings.model,
                    });
                  }}
                >
                  {PROVIDERS.map((p) => (
                    <option key={p.value} value={p.value}>
                      {p.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="settings-section">
                <label htmlFor="model">Model</label>
                <select
                  id="model"
                  value={settings.model}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      model: e.target.value,
                    })
                  }
                >
                  {availableModels.map((m) => (
                    <option key={m} value={m}>
                      {m}
                    </option>
                  ))}
                </select>
              </div>

              {settings.provider === "anthropic" && (
                <div className="settings-section">
                  <label htmlFor="anthropic-key">
                    Anthropic API Key
                  </label>
                  <input
                    id="anthropic-key"
                    type="password"
                    placeholder="sk-ant-..."
                    value={settings.anthropic_api_key || ""}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        anthropic_api_key: e.target.value || undefined,
                      })
                    }
                  />
                </div>
              )}

              {settings.provider === "openai" && (
                <div className="settings-section">
                  <label htmlFor="openai-key">OpenAI API Key</label>
                  <input
                    id="openai-key"
                    type="password"
                    placeholder="sk-..."
                    value={settings.openai_api_key || ""}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        openai_api_key: e.target.value || undefined,
                      })
                    }
                  />
                </div>
              )}

              {settings.provider === "openrouter" && (
                <div className="settings-section">
                  <label htmlFor="openrouter-key">
                    OpenRouter API Key
                  </label>
                  <input
                    id="openrouter-key"
                    type="password"
                    placeholder="sk-or-..."
                    value={settings.openrouter_api_key || ""}
                    onChange={(e) =>
                      setSettings({
                        ...settings,
                        openrouter_api_key: e.target.value || undefined,
                      })
                    }
                  />
                </div>
              )}

              <div className="settings-section">
                <label htmlFor="tavily-key">
                  Tavily API Key (Optional)
                </label>
                <input
                  id="tavily-key"
                  type="password"
                  placeholder="tvly-..."
                  value={settings.tavily_api_key || ""}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      tavily_api_key: e.target.value || undefined,
                    })
                  }
                />
                <small>Optional for web search functionality</small>
              </div>

              {error && <div className="settings-error">{error}</div>}
              {success && (
                <div className="settings-success">Settings saved!</div>
              )}
            </>
          )}
        </div>

        <div className="settings-footer">
          <button
            className="btn-cancel"
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            className="btn-save"
            onClick={handleSave}
            disabled={loading || saving}
          >
            {saving ? "Saving..." : "Save"}
          </button>
        </div>
      </div>
    </div>
  );
}
