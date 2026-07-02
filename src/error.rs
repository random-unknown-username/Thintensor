//! Shared error/report types.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ThinTensorError {
    #[error("verification failed")]
    VerificationFailed,
}

#[derive(Debug, Default, Clone)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Report {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn error(&mut self, message: impl Into<String>) {
        self.errors.push(message.into());
    }

    pub fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }

    pub fn merge(&mut self, other: Report) {
        self.errors.extend(other.errors);
        self.warnings.extend(other.warnings);
    }
}
