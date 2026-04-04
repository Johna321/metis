use schemars::schema_for;
use std::fs;
use std::path::Path;

fn main() {
    let out_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("schemas");
    fs::create_dir_all(&out_dir).expect("failed to create schemas dir");

    let types: Vec<(&str, schemars::schema::RootSchema)> = vec![
        ("IngestResponse", schema_for!(metis_types::IngestResponse)),
        (
            "VectorizeResponse",
            schema_for!(metis_types::VectorizeResponse),
        ),
        ("EvidenceItem", schema_for!(metis_types::EvidenceItem)),
        ("BboxSelection", schema_for!(metis_types::BboxSelection)),
        ("ChatRequest", schema_for!(metis_types::ChatRequest)),
        ("ChatStreamEvent", schema_for!(metis_types::ChatStreamEvent)),
        ("ConversationMeta", schema_for!(metis_types::ConversationMeta)),
        ("ConversationMessage", schema_for!(metis_types::ConversationMessage)),
        ("ConversationFull", schema_for!(metis_types::ConversationFull)),
    ];

    // Write individual schema files
    for (name, schema) in &types {
        let json = serde_json::to_string_pretty(schema).expect("failed to serialize schema");
        let path = out_dir.join(format!("{name}.json"));
        fs::write(&path, json).expect("failed to write schema file");
        println!("wrote {}", path.display());
    }

    // Write a combined schema with all types as definitions
    let mut combined = serde_json::Map::new();
    combined.insert(
        "$schema".into(),
        serde_json::Value::String("http://json-schema.org/draft-07/schema#".into()),
    );

    let mut definitions = serde_json::Map::new();
    for (name, schema) in &types {
        let value = serde_json::to_value(schema).expect("failed to serialize");
        definitions.insert((*name).to_string(), value);
    }
    combined.insert("definitions".into(), serde_json::Value::Object(definitions));

    let combined_json =
        serde_json::to_string_pretty(&combined).expect("failed to serialize combined schema");
    let combined_path = out_dir.join("_combined.json");
    fs::write(&combined_path, combined_json).expect("failed to write combined schema");
    println!("wrote {}", combined_path.display());
}
