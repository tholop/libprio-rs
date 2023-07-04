// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use num_bigint::BigUint;
use num_rational::Ratio;

/// Alias for arbitrary precision unsigned rationals.
pub type BigURational = Ratio<BigUint>;

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Marker trait for differential privacy scalar noise distributions.
pub trait DifferentialPrivacyDistribution {}

/// Zero-concentrated differential privacy (ZCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
pub struct ZeroConcentratedDifferentialPrivacyBudget {
    /// Parameter `epsilon`, using the notation from [[CKS20]] where `rho = (epsilon**2)/2`
    /// for a `rho`-ZCDP budget. A rational number represented as pair of integers.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub epsilon: BigURational,
}

/// Alias for ZeroConcentratedDifferentialPrivacyBudget.
pub type ZCdpBudget = ZeroConcentratedDifferentialPrivacyBudget;

impl DifferentialPrivacyBudget for ZCdpBudget {}
/// Strategy to make aggregate shares differentially private, e.g. by adding noise from a specific
/// type of distribution instantiated with a given DP budget.
pub trait DifferentialPrivacyStrategy {
    /// The type of the DP budget, i.e. the flavour of differential privacy that can be obtained
    /// by using this strategy.
    type Budget: DifferentialPrivacyBudget;
    /// The distribution type this strategy will use to generate the noise.
    type Distribution: DifferentialPrivacyDistribution;
    /// The type the sensitivity used for privacy analysis has.
    type Sensitivity;

    /// Create a strategy from a differential privacy budget. The distribution created with
    /// `create_disctribution` should provide the amount of privacy specified here.
    fn from_budget(b: Self::Budget) -> Self;

    /// Create a new distribution parametrized s.t. adding samples to the result of a function
    /// with sensitivity `s` will yield differential privacy of the flavour given in the
    /// `Budget` type.
    fn create_distribution(&self, s: Self::Sensitivity) -> Self::Distribution;
}

pub mod distributions;

mod experimental_strategy;
