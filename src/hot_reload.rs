use std::collections::HashMap;

use crate::gpu::GpuContext;
use crate::shaders::ShaderEntry;

#[cfg(debug_assertions)]
mod inner {
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

    pub struct ShaderWatcher {
        changed: Arc<AtomicBool>,
        _watcher: RecommendedWatcher,
    }

    impl ShaderWatcher {
        #[allow(clippy::new_without_default)]
        pub fn new() -> Self {
            let changed = Arc::new(AtomicBool::new(false));

            let flag = Arc::clone(&changed);
            let mut watcher =
                notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
                    if let Ok(event) = res
                        && matches!(
                            event.kind,
                            EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
                        )
                    {
                        flag.store(true, Ordering::Relaxed);
                    }
                })
                .expect("Failed to create file watcher");

            watcher
                .watch(Path::new("shaders"), RecursiveMode::Recursive)
                .expect("Failed to watch shaders directory");

            Self {
                changed,
                _watcher: watcher,
            }
        }

        pub fn poll(&mut self) -> bool {
            self.changed.swap(false, Ordering::Relaxed)
        }
    }
}

#[cfg(not(debug_assertions))]
mod inner {
    pub struct ShaderWatcher;

    impl ShaderWatcher {
        pub fn new() -> Self {
            Self
        }

        pub fn poll(&mut self) -> bool {
            false
        }
    }
}

pub use inner::ShaderWatcher;

pub fn reload_shader_module(ctx: &GpuContext, entry: ShaderEntry) -> Option<wgpu::ShaderModule> {
    match entry.create_shader_module_relative_path(&ctx.device, "shaders", HashMap::new(), |path| {
        std::fs::read_to_string(path)
    }) {
        Ok(module) => {
            tracing::info!("Reloaded {:?}", entry);
            Some(module)
        }
        Err(e) => {
            tracing::error!("Shader reload error ({:?}): {e}", entry);
            None
        }
    }
}
