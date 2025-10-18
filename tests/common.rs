use std::path::{Path, PathBuf};

pub async fn download_if_not_exists<S: AsRef<Path>>(filename: S) -> Result<PathBuf, anyhow::Error> {
    if filename.as_ref().exists() {
        println!(
            "File already exists, skipping download: {:?}",
            filename.as_ref().display()
        );
        return Ok(filename.as_ref().to_path_buf());
    }

    let api = hf_hub::api::sync::Api::new()?;
    let filename = api
        .model("rkingzhong/yolov10".to_string())
        .download(filename.as_ref().to_str().unwrap())?;

    Ok(filename)
}
