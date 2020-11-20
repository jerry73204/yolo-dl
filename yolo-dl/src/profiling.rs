use crate::common::*;

#[cfg(feature = "profiling")]
#[derive(Debug)]
pub struct Timing {
    name: &'static str,
    instant: Instant,
    elapsed: Vec<(&'static str, Duration)>,
    non_unicode_once: Once,
}

#[cfg(not(feature = "profiling"))]
#[derive(Debug)]
pub struct Timing;

impl Timing {
    pub fn new(name: &'static str) -> Self {
        #[cfg(feature = "profiling")]
        {
            Self {
                name,
                instant: Instant::now(),
                elapsed: vec![],
                non_unicode_once: Once::new(),
            }
        }

        #[cfg(not(feature = "profiling"))]
        Self
    }

    pub fn set_record<'a>(&mut self, name: &'static str) {
        #[cfg(feature = "profiling")]
        {
            self.elapsed.push((name, self.instant.elapsed()));
            self.instant = Instant::now();
        }
    }

    pub fn report(&self) {
        #[cfg(feature = "profiling")]
        {
            let key = "YOLODL_TIMING";
            let can_report = match env::var(key) {
                Ok(expect_name) => self.name == expect_name,
                Err(VarError::NotPresent) => true,
                Err(VarError::NotUnicode(_)) => {
                    self.non_unicode_once.call_once(|| {
                        error!("the value of '{}' is not unicode", key);
                    });
                    false
                }
            };

            if can_report {
                info!("profiling report for '{}'", self.name);
                self.elapsed.iter().for_each(|(name, elapsed)| {
                    info!("- {}\t{:?}", name, elapsed);
                });
            }
        }
    }
}
