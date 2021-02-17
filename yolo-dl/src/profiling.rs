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

/// A profiler that measures elapsed time between events.
#[cfg(feature = "profiling")]
#[derive(Debug, Clone)]
pub struct Timing {
    name: &'static str,
    instant: Instant,
    root: Option<Arc<Vec<Timing>>>,
    elapsed: Vec<(&'static str, Duration)>,
}

/// A profiler that measures elapsed time between events.
#[cfg(not(feature = "profiling"))]
#[derive(Debug, Clone)]
pub struct Timing;

impl Timing {
    /// Create a new timing profiler.
    pub fn new(name: &'static str) -> Self {
        #[cfg(feature = "profiling")]
        {
            if REGISTERED_TIMINGS.insert(name) {
                info!("registered timing profile '{}'", name);
            }

            Self {
                name,
                instant: Instant::now(),
                root: None,
                elapsed: vec![],
            }
        }

        #[cfg(not(feature = "profiling"))]
        {
            let _ = name;
            Self
        }
    }

    /// Merge from a set of timing profilers with the same name.
    pub fn merge(name: &'static str, timings: Vec<Self>) -> Result<Self> {
        #[cfg(feature = "profiling")]
        {
            ensure!(!timings.is_empty(), "input must not be empty");
            let timing_names: HashSet<_> = timings.iter().map(|timing| timing.name).collect();
            ensure!(
                timing_names.len() == 1,
                "cannot merge from timings with distinct names"
            );
            let timing_name = timing_names.into_iter().next().unwrap();
            let elapsed = timings
                .iter()
                .map(|timing| timing.instant.elapsed())
                .max()
                .unwrap();

            Ok(Self {
                name: timing_name,
                instant: Instant::now(),
                root: Some(Arc::new(timings)),
                elapsed: vec![(name, elapsed)],
            })
        }

        #[cfg(not(feature = "profiling"))]
        {
            let _ = timings;
            let _ = name;
            Ok(Self)
        }
    }

    /// Add an event. It measures the elapsed time since the last event.
    pub fn add_event<'a>(&mut self, name: &'static str) {
        #[cfg(feature = "profiling")]
        {
            self.elapsed.push((name, self.instant.elapsed()));
            self.instant = Instant::now();
        }

        #[cfg(not(feature = "profiling"))]
        let _ = name;
    }

    /// Print report on terminal.
    pub fn report(&self) {
        #[cfg(feature = "profiling")]
        {
            let can_report = PROFILING_CONFIG
                .profiling_whitelist
                .as_ref()
                .map(|whitelist| whitelist.contains(self.name))
                .unwrap_or(true);

            if can_report {
                let mut builder = ptree::TreeBuilder::new(self.name.into());
                self.build_ptree_recursive(&mut builder, &mut Duration::from_secs(0));

                let mut bytes = vec![];
                ptree::write_tree(&builder.build(), &mut bytes).unwrap();
                let tree_text = String::from_utf8(bytes).unwrap();

                info!("profiling report for '{}'\n{}", self.name, tree_text);
            }
        }
    }

    #[cfg(feature = "profiling")]
    fn build_ptree_recursive(
        &self,
        builder: &mut ptree::TreeBuilder,
        elapsed_cumulative: &mut Duration,
    ) {
        if let Some(root) = &self.root {
            *elapsed_cumulative = root
                .iter()
                .enumerate()
                .map(|(index, timing)| {
                    let mut elapsed_cumulative = elapsed_cumulative.clone();

                    builder.begin_child(format!("{}", index));
                    timing.build_ptree_recursive(builder, &mut elapsed_cumulative);
                    builder.end_child();

                    elapsed_cumulative
                })
                .max()
                .unwrap();
        }

        self.elapsed.iter().for_each(|(name, elapsed)| {
            *elapsed_cumulative += *elapsed;
            builder.add_empty_child(format!(
                "- {}\t{:?}\t{:?}",
                name, elapsed, elapsed_cumulative
            ));
        });
    }
}
