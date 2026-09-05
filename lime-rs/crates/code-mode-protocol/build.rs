fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=src/grpc");
    let protoc = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc);
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_protos(&["src/grpc/codex.code_mode.v1.proto"], &["src/grpc"])?;
    Ok(())
}
