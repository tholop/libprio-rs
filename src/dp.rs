// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use num_bigint::{BigUint, TryFromBigIntError};
use num_rational::{BigRational, Ratio};
use num_traits::Unsigned;

/// Errors propagated by methods in this module.
#[derive(Debug, thiserror::Error)]
pub enum DPError {
    /// Tried to use an infinite float as privacy parameter.
    #[error("DP error: input float was infinite.")]
    InfinityFloat(),

    /// Tried to construct a rational number with zero denominator.
    #[error("DP error: input denominator was zero.")]
    ZeroDenominator(),

    /// Tried to convert BigInt into something incompatible.
    #[error("DP error: {0}")]
    BigIntConversion(#[from] TryFromBigIntError<()>),
}

/// Alias for arbitrary precision unsigned rationals.
pub type BigURational = Ratio<BigUint>;

/// Positive finite precision rational number to represent DP and noise distribution parameters in
/// protocol messages and manipulate them without rounding errors.
#[derive(Clone, Debug)]
pub struct URational<T: Unsigned> {
    numerator: T,
    denominator: T,
}

impl<T> URational<T>
where
    T: Unsigned,
{
    /// Construct a `URational` number from numerator and denominator. Errors if denominator is zero.
    pub fn from_unsigned(n: T, d: T) -> Result<Self, DPError> {
        if d.is_zero() {
            Err(DPError::ZeroDenominator())
        } else {
            Ok(URational {
                numerator: n,
                denominator: d,
            })
        }
    }
}

impl<T> From<URational<T>> for BigURational
where
    T: Unsigned + Into<u128>,
{
    // we don't want to expose BigUint to the public API, hence the Into<u128> requirement instead
    fn from(r: URational<T>) -> Self {
        BigURational::new(r.numerator.into().into(), r.denominator.into().into())
    }
}

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Marker trait for differential privacy scalar noise distributions.
pub trait DifferentialPrivacyDistribution {}

/// Zero-concentrated differential privacy (ZCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
pub struct ZeroConcentratedDifferentialPrivacyBudget {
    epsilon: BigURational,
}

/// Alias for ZeroConcentratedDifferentialPrivacyBudget.
pub type ZCdpBudget = ZeroConcentratedDifferentialPrivacyBudget;

impl ZCdpBudget {
    /// Create a budget for parameter `epsilon`, using the notation from [[CKS20]] where `rho = (epsilon**2)/2`
    /// for a `rho`-ZCDP budget.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub fn new<T>(epsilon: URational<T>) -> Self
    where
        T: Unsigned + Into<u128>,
    {
        ZCdpBudget {
            epsilon: epsilon.into(),
        }
    }

    /// Create a budget for parameter `epsilon`, using the notation from [[CKS20]] where `rho = (epsilon**2)/2`
    /// for a `rho`-ZCDP budget. Returns a `DPError` if the input float is not finite and positive.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub fn from_float(epsilon: f32) -> Result<Self, DPError> {
        let eps_rational = match BigRational::from_float(epsilon) {
            Some(y) => {
                BigURational::new(BigUint::try_from(y.numer())?, BigUint::try_from(y.denom())?)
            }
            None => Err(DPError::InfinityFloat())?,
        };
        Ok(ZeroConcentratedDifferentialPrivacyBudget {
            epsilon: eps_rational,
        })
    }
}

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
