use core::fmt;

#[derive(Debug)]
pub enum LoadTemplateError {
    Io(std::io::Error),
}

impl fmt::Display for LoadTemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to load template file: {err}"),
        }
    }
}

impl std::error::Error for LoadTemplateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
        }
    }
}

impl From<std::io::Error> for LoadTemplateError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}
