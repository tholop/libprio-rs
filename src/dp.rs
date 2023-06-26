// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
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

/// Marker trait for differential privacy scalar noise distributions
pub trait DifferentialPrivacyDistribution {}

/// Zero-concentrated differential privacy (zCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
#[allow(dead_code)]
pub struct ZeroConcentratedDifferentialPrivacyBudget {
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

/// Zero-mean Discrete Gaussian noise distribution.
///
/// The distribution is defined over the integers, represented by arbitrary-precision integers.
///
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct DiscreteGaussian {
    sigma: Rational,
}

impl DifferentialPrivacyDistribution for DiscreteGaussian {}

#[allow(unused_variables)]
impl DiscreteGaussian {
    /// Creates a new zero-mean Discrete Gaussian distribution with standard deviation `sigma`.
    pub fn new(sigma: Rational) -> Self {
        Self { sigma }
    }
}

/// Strategy to make aggregate shares differentially private, e.g. by adding noise from a specific
/// type of distribution instantiated with a given DP budget
pub trait DifferentialPrivacyStrategy {}

/// A zCDP budget used to create a Discrete Gaussian distribution
#[allow(dead_code)]
pub struct ZCdpDiscreteGaussian {
    budget: ZeroConcentratedDifferentialPrivacyBudget,
}

impl DifferentialPrivacyStrategy for ZCdpDiscreteGaussian {}

impl ZCdpDiscreteGaussian {
    /// Creates a new Discrete Gaussian by following Theorem 4 from [[CKS20]]
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub fn create_distribution(
        budget: ZeroConcentratedDifferentialPrivacyBudget,
        sensitivity: Rational,
    ) -> DiscreteGaussian {
        let sigma = Rational {
            numerator: budget.epsilon.denominator * sensitivity.numerator,
            denominator: budget.epsilon.numerator * sensitivity.denominator,
        };
        DiscreteGaussian::new(sigma)
    }
}
