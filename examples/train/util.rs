use crate::common::*;

#[derive(Debug)]
pub struct RateCounter {
    closing: Arc<AtomicBool>,
    rate_lock: Arc<RwLock<f64>>,
    curr_lock: Arc<RwLock<f64>>,
    tx: broadcast::Sender<()>,
    rx: broadcast::Receiver<()>,
}

impl RateCounter {
    pub fn new(alpha: f64) -> Self {
        let rate_lock = Arc::new(RwLock::new(0.0));
        let curr_lock = Arc::new(RwLock::new(0.0));
        let closing = Arc::new(AtomicBool::new(false));
        let (tx, rx) = broadcast::channel(1);

        // run ticker
        {
            let tx = tx.clone();
            let mut curr_lock = curr_lock.clone();
            let mut rate_lock = rate_lock.clone();
            let closing = closing.clone();

            async_std::task::spawn(async move {
                {
                    while !closing.load(Ordering::SeqCst) {
                        async_std::task::sleep(Duration::from_secs(1)).await;
                        let mut rate = rate_lock.write().await;
                        let mut curr = curr_lock.write().await;

                        *rate = *curr * alpha + *rate * (1.0 - alpha);
                        *curr = 0.0;

                        if let Err(_) = tx.send(()) {
                            break;
                        }
                    }
                }
            });
        }

        Self {
            rate_lock,
            curr_lock,
            closing,
            tx,
            rx,
        }
    }

    pub async fn add(&self, value: f64) {
        let mut curr = self.curr_lock.write().await;
        *curr += value;
    }

    pub async fn rate(&mut self) -> Option<f64> {
        use broadcast::TryRecvError;

        loop {
            match self.rx.try_recv() {
                Ok(()) => {
                    let rate = *self.rate_lock.read().await;
                    break Some(rate);
                }
                Err(TryRecvError::Closed) | Err(TryRecvError::Empty) => break None,
                Err(TryRecvError::Lagged(_)) => continue,
            }
        }
    }
}

impl Clone for RateCounter {
    fn clone(&self) -> Self {
        let Self {
            closing,
            rate_lock,
            curr_lock,
            tx,
            rx,
        } = self;

        Self {
            closing: closing.clone(),
            rate_lock: rate_lock.clone(),
            curr_lock: curr_lock.clone(),
            tx: tx.clone(),
            rx: tx.subscribe(),
        }
    }
}

impl Drop for RateCounter {
    fn drop(&mut self) {
        self.closing.store(true, Ordering::SeqCst);
    }
}
