// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use crate::field::FieldElement;
use crate::vdaf::prg::SeedStream;
use std::fmt::Debug;

/// Positive rational number to represent DP parameters and noise distributions.
#[allow(dead_code)]
#[derive(Clone, Debug)]
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

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Zero-concentrated differential privacy (zCDP) budget.
#[allow(dead_code)]
struct ZeroConcentratedDifferentialPrivacyBudget {
    epsilon: Rational, // rho = (epsilon**2)/2
}

impl DifferentialPrivacyBudget for ZeroConcentratedDifferentialPrivacyBudget {}

#[allow(dead_code)]
impl ZeroConcentratedDifferentialPrivacyBudget {
    /// Creates a new rho-zCDP budget with parameter `epsilon`, using the
    /// notation from https://arxiv.org/pdf/2004.00010.pdf where `rho = (epsilon**2)/2`.
    pub fn new(epsilon: Rational) -> Self {
        Self { epsilon }
    }
}

/// Trait for a noise distribution over a (vector space over a) finite field.
/// Useful for differential privacy.
/// Inherited by specific mechanism instantiations such as `[DiscreteGaussian]`
/// that will enforce the VDAF-dependent DP guarantees.
pub trait Distribution: Clone + Debug {
    /// Fills the `noise` buffer with elements sampled independently from
    /// the distribution over `F` (i.i.d.) with randomness coming from `seed_stream`.
    fn sample<F: FieldElement, S: SeedStream>(&self, seed_stream: &mut S, noise: &mut [F]);
}

/// Zero-mean Discrete Gaussian noise distribution.
///
/// The distribution is defined over the integers, represented by BigInts.
/// To implement `Distribution`, the `sample` method can project on a finite F if necessary,
/// as long as the DP guarantees are the same as the mechanism over Z
/// (the proof is implementation dependent, e.g. DP post-processing).
///
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct DiscreteGaussian {
    sigma: Rational,
}

#[allow(unused_variables)]
impl DiscreteGaussian {
    /// Creates a new zero-mean Discrete Gaussian distribution with standard deviation `sigma`.
    pub fn new(sigma: Rational) -> Self {
        Self { sigma }
    }
}

impl Distribution for DiscreteGaussian {
    fn sample<F: FieldElement, S: SeedStream>(&self, _seed_stream: &mut S, _noise: &mut [F]) {
        todo!()
    }
}
