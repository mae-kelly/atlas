fn main() {
    cc::Build::new()
        .cpp(true)
        .file("../cpp_core/hft_core.cpp")
        .flag("-std=c++17")
        .flag("-O3")
        .flag("-march=native")
        .flag("-ffast-math")
        .compile("hft_core");
    
    println!("cargo:rerun-if-changed=../cpp_core/hft_core.cpp");
    println!("cargo:rerun-if-changed=../cpp_core/hft_core.hpp");
}
