use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=Qt5Core");
    println!("cargo:rustc-link-lib=Qt5Widgets");
    println!("cargo:rustc-link-lib=Qt5Gui");
    println!("cargo:rustc-link-lib=qcustomplot");
    
    // Detect Qt installation
    let qt_dir = env::var("QT_DIR").unwrap_or_else(|_| {
        if cfg!(target_os = "windows") {
            "C:/Qt/5.15.2/msvc2019_64".to_string()
        } else if cfg!(target_os = "macos") {
            "/usr/local/opt/qt5".to_string()
        } else {
            "/usr/include/qt5".to_string()
        }
    });
    
    println!("cargo:rustc-link-search={}/lib", qt_dir);
    
    // Configure Qt binding generation
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", qt_dir))
        .allowlist_type("^Q.*")
        .allowlist_function("^q.*")
        .generate()
        .expect("Unable to generate Qt bindings");
    
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("qt_bindings.rs"))
        .expect("Couldn't write Qt bindings!");
    
    // Run Qt's meta-object compiler (moc)
    let status = Command::new(format!("{}/bin/moc", qt_dir))
        .args(&["src/qt_integration.rs", "-o", "src/moc_qt_integration.rs"])
        .status()
        .expect("Failed to execute Qt MOC");
    
    if !status.success() {
        panic!("Qt MOC failed to compile qt_integration.rs");
    }
    
    println!("cargo:rerun-if-changed=src/qt_integration.rs");
    println!("cargo:rerun-if-changed=wrapper.h");
} 