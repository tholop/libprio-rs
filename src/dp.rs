// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use crate::field::FieldElement;
use crate::vdaf::prg::SeedStream;
use std::fmt::Debug;

// TODO(tholop): We could also use pairs of integers, or even
// floats to store the DiscreteGaussian DP parameters?
use num_rational::BigRational;

/// Trait for a noise distribution over a (vector space over a) finite field.
/// Useful for differential privacy.
/// Inherited by specific mechanism instantiations such as `[DiscreteGaussian]`
/// that will enforce the VDAF-dependent DP guarantees.
pub trait Dist: Clone + Debug {
    /// Sample `len` elements from the distribution over `F`.
    /// The randomness comes from a Prg seeded by `seed`.
    /// TODO(tholop): do we need a generic SEED_SIZE here, or 16 is fine?
    fn sample<F: FieldElement, S: SeedStream>(&self, seed_stream: S, len: usize) -> Vec<F>;
}

/// Discrete Gaussian noise distribution with mean 0 and standard deviation `sigma`.
/// `sigma` can be computed as a function of DP parameters `epsilon`, `delta`
/// and `base_sensitivity`. `base_sensitivity` depends on the VDAF but not
/// on the number of measurements. The `Aggregator` is in charge of scaling
/// the noise by a function of the number of measurements, if necessary.
/// `sigma` is sufficient to characterize the distribution, but other parameters
/// are kept for interpretability.
///
/// The distribution is defined over the integers, represented by BigInts
/// To implement `Dist`, the `sample` method can project on a finite F if necessary,
/// as long as the DP guarantees are the same as the mechanism over Z
/// (the proof is implementation dependent, e.g. DP post-processing)
///
#[allow(dead_code)]
pub struct DiscreteGaussian {
    epsilon: BigRational,
    delta: BigRational,
    sigma: BigRational,
    base_sensitivity: BigRational,
}

#[allow(unused_variables)]
impl DiscreteGaussian {
    /// Create a new DiscreteGaussian distribution with the given DP parameters.
    pub fn new(epsilon: BigRational, delta: BigRational, sensitivity: BigRational) -> Self {
        todo!();
    }
}
