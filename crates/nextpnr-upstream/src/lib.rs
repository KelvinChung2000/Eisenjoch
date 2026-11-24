use std::ffi::{c_char, c_int, CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr;

use libloading::Library;
use tempfile::NamedTempFile;

#[derive(Clone, Copy, Debug)]
pub enum Placer {
    Heap,
    Sa,
}

#[derive(Clone, Copy, Debug)]
pub enum Router {
    Router1,
    Router2,
}

#[derive(Clone, Copy, Debug, Default)]
#[repr(u32)]
pub enum Stage {
    Pack = 0,
    Place = 1,
    #[default]
    Route = 2,
}

pub struct Request<'a> {
    pub device: &'a str,
    pub chipdb: &'a Path,
    pub input: &'a Path,
    pub output: &'a Path,
    pub placer: Option<Placer>,
    pub router: Option<Router>,
    pub seed: u64,
    pub stop_after: Stage,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot resolve nextpnr library {path:?}: {source}")]
    LibraryPath {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("cannot load nextpnr bridge {path:?}: {source}")]
    Library {
        path: PathBuf,
        source: libloading::Error,
    },
    #[error("{0} contains a NUL byte")]
    InvalidString(&'static str),
    #[error("upstream nextpnr failed: {0}")]
    Nextpnr(String),
    #[error("cannot write nextpnr output {path:?}: {source}")]
    Output {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[repr(C)]
struct NativeRequest {
    device: *const c_char,
    chipdb: *const c_char,
    input: *const c_char,
    output: *const c_char,
    placer: *const c_char,
    router: *const c_char,
    seed: u64,
    stop_after: Stage,
}

type Run = unsafe extern "C" fn(*const NativeRequest, *mut c_char, usize) -> c_int;

pub struct Nextpnr {
    run: Run,
    _library: Library,
}

impl Nextpnr {
    /// # Safety
    /// The library must be trusted and implement this crate's versioned bridge ABI.
    pub unsafe fn load(path: &Path) -> Result<Self, Error> {
        let resolved = path.canonicalize().map_err(|source| Error::LibraryPath {
            path: path.to_owned(),
            source,
        })?;
        let load_error = |source| Error::Library {
            path: resolved.clone(),
            source,
        };
        let library = unsafe { Library::new(&resolved) }.map_err(load_error)?;
        let run = *unsafe { library.get::<Run>(c"eisenjoch_nextpnr_run_v1".to_bytes_with_nul()) }
            .map_err(load_error)?;
        Ok(Self {
            run,
            _library: library,
        })
    }

    /// # Safety
    /// The chip database must be valid for the device and the loaded nextpnr
    /// revision. Its contents must remain immutable for the duration of the call.
    pub unsafe fn run(&self, request: &Request<'_>) -> Result<(), Error> {
        let device = CString::new(request.device).map_err(|_| Error::InvalidString("device"))?;
        let chipdb = path_to_c_string(request.chipdb, "chipdb path")?;
        let input = path_to_c_string(request.input, "input path")?;
        let output_error = |source| Error::Output {
            path: request.output.to_owned(),
            source,
        };
        let output_dir = match request.output.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => parent,
            _ => Path::new("."),
        };
        let temporary = NamedTempFile::new_in(output_dir).map_err(output_error)?;
        let output = path_to_c_string(temporary.path(), "temporary output path")?;
        let placer = match request.placer {
            Some(Placer::Heap) => c"heap".as_ptr(),
            Some(Placer::Sa) => c"sa".as_ptr(),
            None => ptr::null(),
        };
        let router = match request.router {
            Some(Router::Router1) => c"router1".as_ptr(),
            Some(Router::Router2) => c"router2".as_ptr(),
            None => ptr::null(),
        };
        let native = NativeRequest {
            device: device.as_ptr(),
            chipdb: chipdb.as_ptr(),
            input: input.as_ptr(),
            output: output.as_ptr(),
            placer,
            router,
            seed: request.seed,
            stop_after: request.stop_after,
        };
        let mut error = [0u8; 4096];
        let status = unsafe { (self.run)(&native, error.as_mut_ptr().cast(), error.len()) };
        if status != 0 {
            let message =
                CStr::from_bytes_until_nul(&error).expect("bridge must terminate its diagnostic");
            return Err(Error::Nextpnr(message.to_string_lossy().trim().to_owned()));
        }
        temporary
            .persist(request.output)
            .map_err(|failure| output_error(failure.error))?;
        Ok(())
    }
}

fn path_to_c_string(path: &Path, field: &'static str) -> Result<CString, Error> {
    CString::new(path.as_os_str().as_encoded_bytes()).map_err(|_| Error::InvalidString(field))
}
