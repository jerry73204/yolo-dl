use crate::common::*;

/// The learning rate scheduling strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Config {
    /// Use constant learning rate.
    Constant { lr: R64 },
    /// Use specific learning rate at specified steps.
    StepWise { steps: Vec<(usize, R64)> },
}

/// Learning rate scheduling algorithm.
#[derive(Debug, Clone)]
pub enum LrScheduler {
    Constant {
        lr: R64,
    },
    StepWise {
        lr_cache: f64,
        step: usize,
        index: usize,
        steps: Vec<(usize, R64)>,
    },
}

impl LrScheduler {
    pub fn new(config: &Config, init_step: impl Into<Option<usize>>) -> Result<Self> {
        let init_step = init_step.into();

        let mut scheduler = match *config {
            Config::Constant { lr } => {
                ensure!(lr >= 0.0, "the lr must be positive");
                Self::Constant { lr }
            }
            Config::StepWise { ref steps } => {
                ensure!(
                    !steps.is_empty() && steps[0].0 == 0,
                    "the steps must start from zero"
                );

                steps
                    .iter()
                    .try_fold(None, |prev_step, (curr_step, lr)| -> Result<_> {
                        if let Some(prev_step) = prev_step {
                            ensure!(curr_step > prev_step, "the steps must be monotonic");
                        }
                        ensure!(lr.raw() > 0.0, "learning rate must be positive");
                        Ok(Some(curr_step))
                    })?;

                Self::StepWise {
                    lr_cache: steps[0].1.raw(),
                    step: 0,
                    index: 0,
                    steps: steps.clone(),
                }
            }
        };

        if let Some(init_step) = init_step {
            scheduler.set_step(init_step);
        }

        Ok(scheduler)
    }

    pub fn set_step(&mut self, new_step: usize) {
        if let Self::StepWise {
            step,
            index,
            steps,
            lr_cache,
        } = self
        {
            *step = new_step;
            let new_index =
                match steps.binary_search_by_key(&new_step, |(step_thresh, _lr)| *step_thresh) {
                    Ok(new_index) => new_index,
                    Err(new_index) => {
                        if new_index > 0 {
                            new_index - 1
                        } else {
                            new_index
                        }
                    }
                };
            *index = new_index;
            *lr_cache = steps[new_index].1.raw();
        }
    }

    pub fn lr(&self) -> f64 {
        match self {
            Self::Constant { lr } => lr.raw(),
            Self::StepWise { lr_cache, .. } => *lr_cache,
        }
    }

    pub fn next(&mut self) -> f64 {
        match self {
            Self::Constant { lr } => lr.raw(),
            Self::StepWise {
                step,
                index,
                steps,
                lr_cache,
            } => {
                let lr = steps[*index].1.raw();
                *step += 1;
                let next_index = *index + 1;
                if next_index < steps.len() && *step == steps[next_index].0 {
                    *index = next_index;
                }
                *lr_cache = lr;
                lr
            }
        }
    }
}
