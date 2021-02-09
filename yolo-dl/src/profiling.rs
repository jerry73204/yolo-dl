//! The profiling toolkit that can be disabled in compile-time.

use crate::common::*;

lazy_static! {
    static ref PROFILING_CONFIG: ProfilingConfig = {
        let config: ProfilingConfig = match envy::prefixed("YOLODL_").from_env() {
            Ok(config) => config,
            Err(err) => {
                eprintln!("failed to load profiling environment variables, fallback to default values: {:?}", err);
                Default::default()
            }
        };
        config
    };
    static ref REGISTERED_TIMINGS: DashSet<&'static str> = DashSet::new();
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub profiling_whitelist: Option<HashSet<String>>,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            profiling_whitelist: None,
        }
    }
}

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
            if REGISTERED_TIMINGS.insert(name) {
                info!("registered timing profile '{}'", name);
            }

            Self {
                name,
                instant: Instant::now(),
                elapsed: vec![],
                non_unicode_once: Once::new(),
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            let _ = name;
            Self
        }
    }

    pub fn set_record<'a>(&mut self, name: &'static str) {
        #[cfg(feature = "profiling")]
        {
            self.elapsed.push((name, self.instant.elapsed()));
            self.instant = Instant::now();
        }

        #[cfg(not(feature = "profiling"))]
        let _ = name;
    }

    pub fn report(&self) {
        #[cfg(feature = "profiling")]
        {
            let can_report = PROFILING_CONFIG
                .profiling_whitelist
                .as_ref()
                .map(|whitelist| whitelist.contains(self.name))
                .unwrap_or(true);

            if can_report {
                info!("profiling report for '{}'", self.name);
                self.elapsed.iter().for_each(|(name, elapsed)| {
                    info!("- {}\t{:?}", name, elapsed);
                });
            }
        }
    }
}
