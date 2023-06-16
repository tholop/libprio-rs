// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use crate::field::FieldElement;
use crate::vdaf::prg::Seed;
use std::fmt::Debug;

// TODO(tholop): We could also use pairs of integers, or even
// floats to store the DiscreteGaussian DP parameters?
// Using BigInts might make it harder to communicate `DiscreteGaussian` to the Collector.
use num_rational::BigRational;

/// DP-related errors.
#[derive(Debug, thiserror::Error)]
pub enum DpError {
    /// Error while sampling from a specific noise distribution
    #[error("sampling error")]
    SamplingError,
}

/// Trait for a noise distribution over a (vector space over a) finite field.
/// Useful for differential privacy.
/// Inherited by specific mechanism instantiations such as `[DiscreteGaussian]`
/// that will enforce the VDAF-dependent DP guarantees.
pub trait Dist: Clone + Debug {
    /// Sample `len` elements from the distribution over `F`.
    /// The randomness comes from a Prg seeded by `seed`.
    /// TODO(tholop): do we need a generic SEED_SIZE here, or 16 is fine?
    fn sample<F: FieldElement, const SEED_SIZE: usize>(
        &self,
        seed: Seed<SEED_SIZE>,
        len: usize,
    ) -> Result<Vec<F>, DpError>;

    // TODO(tholop): do we need another method to send the distribution parameters
    // to the Collector?
}

/// Discrete Gaussian noise distribution with mean 0 and standard deviation `sigma`.
/// `sigma` can be computed as a function of DP parameters `epsilon`, `delta`
/// and `sensitivity`. `sensitivity` depends on the VDAF and potentially the number of measurements.
/// `sigma` is sufficient to characterize the distribution, but other parameters
/// are kept for interpretability.
///
/// The distribution is defined over Z, represented by BigInts
/// To implement `Dist`, the `sample` method can project on a finite F if necessary,
/// as long as the DP guarantees are the same as the mechanism over Z
/// (the proof is implementation dependent, e.g. DP post-processing)
///
/// TODO(tholop): depending on how we handle `num_measurements`, rename `sensitivity`
/// to something like `base_sensitivity` or `sensitivity_when_num_measurements_is_1`,
/// and explain that `add_noise_to_agg_share` is in charge of scaling the noise properly.
pub struct DiscreteGaussian {
    epsilon: BigRational,
    delta: BigRational,
    sigma: BigRational,
    sensitivity: BigRational,
}

impl DiscreteGaussian {
    pub fn new(epsilon: BigRational, delta: BigRational, sensitivity: BigRational) -> Self {
        let sigma = todo!();
    }
}
