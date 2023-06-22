// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use rand::distributions::Distribution;
use std::fmt::Debug;

/// Positive rational number to represent DP and noise distribution parameters in protocol messages
/// and manipulate them without rounding errors.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Rational {
    numerator: u32,
    denominator: u32,
}

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Zero-concentrated differential privacy (zCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
#[allow(dead_code)]
struct ZeroConcentratedDifferentialPrivacyBudget {
    epsilon: Rational,
}

impl DifferentialPrivacyBudget for ZeroConcentratedDifferentialPrivacyBudget {}

#[allow(dead_code)]
impl ZeroConcentratedDifferentialPrivacyBudget {
    /// Creates a new rho-zCDP budget with parameter `epsilon`, using the notation from [[CKS20]]
    /// where `rho = (epsilon**2)/2`.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub fn new(epsilon: Rational) -> Self {
        Self { epsilon }
    }
}

/// Distribution over `T` that can be instantiated from a given privacy budget
pub trait DifferentialPrivacyDistribution<DPBudget: DifferentialPrivacyBudget, T>:
    Distribution<T>
{
    /// Creates a new distribution over `T` such that the additive noising mechanism is
    /// `privacy_budget`-DP when applied to a function with sensitivity `sensitivity`.
    fn from(privacy_budget: &DPBudget, sensitivity: Rational) -> Self;
}

/// Zero-mean Discrete Gaussian noise distribution.
///
/// The distribution is defined over the integers, represented by arbitrary-precision integers.
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
