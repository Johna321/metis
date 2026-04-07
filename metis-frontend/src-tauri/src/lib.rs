use futures::StreamExt;
use metis_types::{
    BboxSelection, ChatStreamEvent, EvidenceItem, IngestResponse, VectorizeResponse,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Mutex;
use tauri::{Emitter, Manager};
use tauri_plugin_shell::{
    process::{CommandChild, CommandEvent},
    ShellExt,
};

struct SidecarHandle {
    child: Mutex<Option<CommandChild>>,
    pid: u32,
}

const BACKEND_URL: &str = "http://127.0.0.1:8000";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiSettings {
    pub provider: String,
    pub model: String,
    pub anthropic_api_key: Option<String>,
    pub openai_api_key: Option<String>,
    pub openrouter_api_key: Option<String>,
    pub tavily_api_key: Option<String>,
}

impl Default for ApiSettings {
    fn default() -> Self {
        Self {
            provider: "anthropic".to_string(),
            model: "claude-sonnet-4-20250514".to_string(),
            anthropic_api_key: None,
            openai_api_key: None,
            openrouter_api_key: None,
            tavily_api_key: None,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MetisError {
    #[error("Document not found: {0}")]
    NotFound(String),
    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),
    #[error("Ingest failed: {0}")]
    IngestFailed(String),
    #[error("Vectorize failed: {0}")]
    VectorizeFailed(String),
    #[error("LLM error: {0}")]
    LlmError(String),
}

impl Serialize for MetisError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

fn map_reqwest_err(e: reqwest::Error) -> MetisError {
    MetisError::BackendUnavailable(e.to_string())
}

/// Compute the SHA256 doc_id for a PDF file without uploading it.
/// Returns "sha256:<hex>" — identical to what the backend would compute.
#[tauri::command]
fn hash_file(file_path: String) -> Result<String, MetisError> {
    let bytes = std::fs::read(&file_path)
        .map_err(|e| MetisError::IngestFailed(format!("Cannot read file: {e}")))?;
    let hash = Sha256::digest(&bytes);
    Ok(format!("sha256:{}", hex::encode(hash)))
}

/// Check whether a document has already been ingested (spans exist in the backend store).
/// Returns the stored IngestResponse if found, or None if not yet ingested.
#[tauri::command]
async fn check_doc(doc_id: String) -> Result<Option<IngestResponse>, MetisError> {
    let resp = reqwest::Client::new()
        .get(format!("{BACKEND_URL}/documents/{doc_id}"))
        .send()
        .await
        .map_err(map_reqwest_err)?;

    if resp.status() == 404 {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(MetisError::IngestFailed(
            resp.text().await.unwrap_or_default(),
        ));
    }

    let meta = resp
        .json::<IngestResponse>()
        .await
        .map_err(|e| MetisError::IngestFailed(e.to_string()))?;
    Ok(Some(meta))
}

#[tauri::command]
async fn ingest_pdf(file_path: String) -> Result<IngestResponse, MetisError> {
    let file_name = std::path::Path::new(&file_path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file.pdf".into());

    let bytes = std::fs::read(&file_path)
        .map_err(|e| MetisError::IngestFailed(format!("Cannot read file: {e}")))?;

    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name(file_name)
        .mime_str("application/pdf")
        .map_err(|e| MetisError::IngestFailed(e.to_string()))?;

    let form = reqwest::multipart::Form::new().part("file", part);

    let resp = reqwest::Client::new()
        .post(format!("{BACKEND_URL}/ingest"))
        .multipart(form)
        .send()
        .await
        .map_err(map_reqwest_err)?;

    if resp.status() == 404 {
        return Err(MetisError::NotFound(resp.text().await.unwrap_or_default()));
    }
    if !resp.status().is_success() {
        return Err(MetisError::IngestFailed(
            resp.text().await.unwrap_or_default(),
        ));
    }

    resp.json::<IngestResponse>()
        .await
        .map_err(|e| MetisError::IngestFailed(e.to_string()))
}

#[tauri::command]
async fn vectorize_doc(doc_id: String) -> Result<VectorizeResponse, MetisError> {
    let body = serde_json::json!({ "doc_id": doc_id });

    let resp = reqwest::Client::new()
        .post(format!("{BACKEND_URL}/vectorize"))
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest_err)?;

    if resp.status() == 404 {
        return Err(MetisError::NotFound(resp.text().await.unwrap_or_default()));
    }
    if !resp.status().is_success() {
        return Err(MetisError::VectorizeFailed(
            resp.text().await.unwrap_or_default(),
        ));
    }

    resp.json::<VectorizeResponse>()
        .await
        .map_err(|e| MetisError::VectorizeFailed(e.to_string()))
}

#[tauri::command]
async fn retrieve_evidence(
    doc_id: String,
    page: u32,
    selected_text: String,
) -> Result<Vec<EvidenceItem>, MetisError> {
    let body = serde_json::json!({
        "doc_id": doc_id,
        "page": page,
        "selected_text": selected_text,
    });

    let resp = reqwest::Client::new()
        .post(format!("{BACKEND_URL}/retrieve"))
        .json(&body)
        .send()
        .await
        .map_err(map_reqwest_err)?;

    if resp.status() == 404 {
        return Err(MetisError::NotFound(resp.text().await.unwrap_or_default()));
    }
    if !resp.status().is_success() {
        return Err(MetisError::BackendUnavailable(
            resp.text().await.unwrap_or_default(),
        ));
    }

    resp.json::<Vec<EvidenceItem>>()
        .await
        .map_err(|e| MetisError::BackendUnavailable(e.to_string()))
}

#[tauri::command]
fn get_document_pdf_url(doc_id: String) -> String {
    format!("{BACKEND_URL}/documents/{doc_id}/pdf")
}

#[tauri::command]
async fn chat_start(
    app: tauri::AppHandle,
    doc_id: String,
    message: String,
    provider: Option<String>,
    model: Option<String>,
    selections: Option<Vec<BboxSelection>>,
) -> Result<(), MetisError> {
    let mut body = serde_json::json!({
        "doc_id": doc_id,
        "message": message,
    });
    if let Some(p) = provider {
        body["provider"] = serde_json::Value::String(p);
    }
    if let Some(m) = model {
        body["model"] = serde_json::Value::String(m);
    }
    if let Some(s) = selections {
        body["selections"] = serde_json::to_value(s).unwrap_or_default();
    }

    // Spawn the streaming task so invoke returns immediately
    tauri::async_runtime::spawn(async move {
        let emit_error = |app: &tauri::AppHandle, msg: String| {
            let _ = app.emit("chat-stream", ChatStreamEvent::Error { message: msg });
        };

        let resp = match reqwest::Client::new()
            .post(format!("{BACKEND_URL}/chat"))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => r,
            Err(e) => {
                emit_error(&app, format!("Backend unavailable: {e}"));
                return;
            }
        };

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            emit_error(&app, format!("Chat request failed: {text}"));
            return;
        }

        // Stream the response body chunk-by-chunk, accumulating lines
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut current_event: Option<String> = None;

        while let Some(chunk_result) = stream.next().await {
            let chunk = match chunk_result {
                Ok(c) => c,
                Err(e) => {
                    emit_error(&app, format!("Stream error: {e}"));
                    return;
                }
            };

            buf.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines
            while let Some(newline_pos) = buf.find('\n') {
                let line = buf[..newline_pos].trim_end_matches('\r').to_string();
                buf = buf[newline_pos + 1..].to_string();

                if line.is_empty() {
                    // End of SSE block
                    current_event = None;
                    continue;
                }

                if let Some(event_name) = line.strip_prefix("event: ") {
                    current_event = Some(event_name.to_string());
                } else if let Some(data_str) = line.strip_prefix("data: ") {
                    let event_name = match &current_event {
                        Some(n) => n.as_str(),
                        None => continue,
                    };

                    let parsed: serde_json::Value = match serde_json::from_str(data_str) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };

                    let stream_event = match event_name {
                        "text_delta" => {
                            let text = parsed["text"].as_str().unwrap_or_default().to_string();
                            ChatStreamEvent::TextDelta { text }
                        }
                        "tool_call_start" => {
                            let name = parsed["name"].as_str().unwrap_or_default().to_string();
                            ChatStreamEvent::ToolCallStart { name }
                        }
                        "tool_call_delta" => {
                            let text = parsed["text"].as_str().unwrap_or_default().to_string();
                            ChatStreamEvent::ToolCallDelta { text }
                        }
                        "tool_call_done" => {
                            let id = parsed["id"].as_str().unwrap_or_default().to_string();
                            let name = parsed["name"].as_str().unwrap_or_default().to_string();
                            let arguments = parsed["arguments"].clone();
                            ChatStreamEvent::ToolCallDone {
                                id,
                                name,
                                arguments,
                            }
                        }
                        "message_done" => {
                            let role = parsed["role"].as_str().unwrap_or_default().to_string();
                            let content = parsed["content"].as_str().map(String::from);
                            let tool_calls = parsed.get("tool_calls").cloned();
                            ChatStreamEvent::MessageDone {
                                role,
                                content,
                                tool_calls,
                            }
                        }
                        "citation_data" => {
                            let tool_call_id = parsed["tool_call_id"]
                                .as_str()
                                .unwrap_or_default()
                                .to_string();
                            let tool_name =
                                parsed["tool_name"].as_str().unwrap_or_default().to_string();
                            let items: Vec<EvidenceItem> =
                                serde_json::from_value(parsed["items"].clone()).unwrap_or_default();
                            ChatStreamEvent::CitationData {
                                tool_call_id,
                                tool_name,
                                items,
                            }
                        }
                        "agent_done" => ChatStreamEvent::AgentDone,
                        "error" => {
                            let message = parsed["message"]
                                .as_str()
                                .or_else(|| parsed["detail"].as_str())
                                .unwrap_or("Unknown error")
                                .to_string();
                            ChatStreamEvent::Error { message }
                        }
                        _ => continue,
                    };

                    let _ = app.emit("chat-stream", stream_event);
                }
            }
        }
    });

    Ok(())
}

fn settings_path(config_dir: &Path) -> std::path::PathBuf {
    config_dir.join("settings.json")
}

fn read_settings_from_file(config_dir: &Path) -> ApiSettings {
    let path = settings_path(config_dir);
    std::fs::read(&path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn write_settings_to_file(config_dir: &Path, settings: &ApiSettings) -> Result<(), MetisError> {
    std::fs::create_dir_all(config_dir)
        .map_err(|e| MetisError::BackendUnavailable(format!("Cannot create config dir: {e}")))?;
    let json = serde_json::to_vec_pretty(settings)
        .map_err(|e| MetisError::BackendUnavailable(format!("Cannot serialize settings: {e}")))?;
    std::fs::write(settings_path(config_dir), json)
        .map_err(|e| MetisError::BackendUnavailable(format!("Cannot write settings file: {e}")))?;
    Ok(())
}

async fn push_settings_to_backend(settings: &ApiSettings) -> Result<(), reqwest::Error> {
    reqwest::Client::new()
        .put(format!("{BACKEND_URL}/settings"))
        .json(settings)
        .send()
        .await?
        .error_for_status()?;
    Ok(())
}

#[tauri::command]
async fn read_settings(app: tauri::AppHandle) -> Result<ApiSettings, MetisError> {
    let config_dir = app
        .path()
        .app_config_dir()
        .map_err(|e| MetisError::BackendUnavailable(e.to_string()))?;
    Ok(read_settings_from_file(&config_dir))
}

#[tauri::command]
async fn save_settings(app: tauri::AppHandle, settings: ApiSettings) -> Result<(), MetisError> {
    let config_dir = app
        .path()
        .app_config_dir()
        .map_err(|e| MetisError::BackendUnavailable(e.to_string()))?;

    write_settings_to_file(&config_dir, &settings)?;

    // Push the updated settings to the running backend so they take effect immediately.
    push_settings_to_backend(&settings)
        .await
        .map_err(|e| MetisError::BackendUnavailable(format!("Failed to sync settings to backend: {e}")))?;

    Ok(())
}

/// App entry point
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let (mut rx, child) = app
                .shell()
                .sidecar("metis-web")
                .expect("failed to create metis-web sidecar command")
                .spawn()
                .expect("failed to spawn metis-web sidecar");

            let pid = child.pid();
            app.manage(SidecarHandle {
                child: Mutex::new(Some(child)),
                pid,
            });

            tauri::async_runtime::spawn(async move {
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) => {
                            print!("[metis-web] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Stderr(line) => {
                            eprint!("[metis-web] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Terminated(status) => {
                            eprint!("[metis-web] terminated: {:?}", status);
                            break;
                        }
                        _ => {}
                    }
                }
            });

            // Push stored settings to the backend once it's ready.
            // Polls every 500 ms for up to 15 seconds before giving up.
            let config_dir = app.path().app_config_dir()?;
            let startup_settings = read_settings_from_file(&config_dir);
            tauri::async_runtime::spawn(async move {
                for _ in 0..30 {
                    if push_settings_to_backend(&startup_settings).await.is_ok() {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            hash_file,
            check_doc,
            ingest_pdf,
            vectorize_doc,
            retrieve_evidence,
            get_document_pdf_url,
            chat_start,
            read_settings,
            save_settings,
        ])
        .build(tauri::generate_context!())
        .expect("error while building Tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                let state = app.state::<SidecarHandle>();

                #[cfg(unix)]
                let _ = std::process::Command::new("kill")
                    .args(["-15", &state.pid.to_string()])
                    .status();

                #[cfg(windows)]
                let _ = std::process::Command::new("taskkill")
                    .args(["/F", "/T", "/PID", &state.pid.to_string()])
                    .status();

                // Wait a little, then send SIGKILL to ensure process is dead
                std::thread::sleep(std::time::Duration::from_millis(500));
                if let Ok(mut guard) = state.child.lock() {
                    if let Some(child) = guard.take() {
                        let _ = child.kill();
                    }
                };
            }
        });
}
