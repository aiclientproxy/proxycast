fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/grpc");
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["src/grpc/codex.code_mode.v1.proto"], &["src/grpc"])?;
    Ok(())
}
