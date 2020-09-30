use crate::common::*;

#[derive(Debug, Clone)]
pub struct Timing {
    instant: Instant,
    elapsed: Vec<(String, Duration)>,
}

impl Timing {
    pub fn new() -> Self {
        Self {
            instant: Instant::now(),
            elapsed: vec![],
        }
    }

    pub fn set_record<'a>(&mut self, name: impl Into<Cow<'a, str>>) {
        self.elapsed
            .push((name.into().into_owned(), self.instant.elapsed()));
        self.instant = Instant::now();
    }

    pub fn records(&self) -> &[(String, Duration)] {
        &self.elapsed
    }
}
