// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use crate::field::FieldElement;
use crate::vdaf::prg::SeedStream;
use std::fmt::Debug;

/// Positive rational number to represent DP parameters and noise distributions.
#[allow(dead_code)]
pub struct Rational {
    numerator: u32,
    denominator: u32,
}

impl Rational {
    #[allow(dead_code)]
    fn new(numerator: u32, denominator: u32) -> Self {
        Rational {
            numerator,
            denominator,
        }
    }
}
/// Trait for a noise distribution over a (vector space over a) finite field.
/// Useful for differential privacy.
/// Inherited by specific mechanism instantiations such as `[DiscreteGaussian]`
/// that will enforce the VDAF-dependent DP guarantees.
pub trait Distribution: Clone + Debug {
    /// Fills the `noise` buffer with elements from the distribution over `F`,
    /// with randomness coming from `seed_stream`
    fn sample<F: FieldElement, S: SeedStream>(&self, seed_stream: S, noise: &mut [F]) -> Vec<F>;
}

/// Discrete Gaussian noise distribution.
///
/// The distribution is defined over the integers, represented by BigInts
/// To implement `Distribution`, the `sample` method can project on a finite F if necessary,
/// as long as the DP guarantees are the same as the mechanism over Z
/// (the proof is implementation dependent, e.g. DP post-processing)
///
#[allow(dead_code)]
pub struct DiscreteGaussian {
    epsilon: Rational,
    delta: Rational,
    sigma: Rational,
}

#[allow(unused_variables)]
impl DiscreteGaussian {
    /// Create a new Discrete Gaussian distribution with mean 0 and
    /// standard deviation such that the additive mechanism is
    /// (`epsilon`, `delta`)-DP for functions with L2 sensivity 1.
    ///
    /// NOTE: we might want to provide a CDP constructor too
    /// See https://arxiv.org/pdf/2004.00010.pdf, Thm 4 for CDP, Thm 7 for Approximate DP.
    ///
    /// `sigma` is sufficient to characterize the distribution, but other parameters
    /// can be kept in the struct for interpretability.
    pub fn new(epsilon: Rational, delta: Rational, sensitivity: Rational) -> Self {
        todo!();
    }
}
