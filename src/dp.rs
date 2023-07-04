// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
#[cfg(feature = "experimental")]
use crate::dp::samplers::DiscreteGaussian;
#[cfg(feature = "experimental")]
use num_bigint::BigUint;
#[cfg(feature = "experimental")]
use num_rational::Ratio;

#[cfg(feature = "experimental")]
mod samplers;

#[cfg(feature = "experimental")]
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
    /// for a `rho`-ZCDP budget.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub epsilon: BigURational,
}

type ZCDPBudget = ZeroConcentratedDifferentialPrivacyBudget;

impl DifferentialPrivacyBudget for ZCDPBudget {}

#[cfg(feature = "experimental")]
impl DifferentialPrivacyDistribution for DiscreteGaussian {}

/// A DP strategy using the discrete gaussian distribution.
pub struct DPDiscreteGaussian<B>
where
    B: DifferentialPrivacyBudget,
{
    budget: B,
}

/// A DP strategy using the discrete gaussian distribution providing zero-concentrated DP.
pub type ZCdpDiscreteGaussian = DPDiscreteGaussian<ZeroConcentratedDifferentialPrivacyBudget>;

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

#[cfg(feature = "experimental")]
impl DifferentialPrivacyStrategy for DPDiscreteGaussian<ZCDPBudget> {
    type Budget = ZCDPBudget;
    type Distribution = DiscreteGaussian;
    type Sensitivity = BigURational;

    fn from_budget(b: ZCDPBudget) -> DPDiscreteGaussian<ZCDPBudget> {
        DPDiscreteGaussian { budget: b }
    }
    /// Create a new sampler from the Discrete Gaussian Distribution with a standard
    /// deviation calibrated to provide `1/2 epsilon^2` zero-concentrated differential
    /// privacy when added to the result of an interger-valued function with sensitivity
    /// `sensitivity`, following Theorem 4 from [[CKS20]]
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    fn create_distribution(&self, sensitivity: BigURational) -> DiscreteGaussian {
        DiscreteGaussian::new(sensitivity / self.budget.epsilon.clone())
    }
}

#[cfg(test)]
mod tests {

    use rand::distributions::Distribution;

    use crate::vdaf::prg::{Seed, SeedStreamSha3};

    use super::*;
    use num_bigint::BigUint;
    use rand::SeedableRng;

    #[test]
    fn test_discrete_gaussian() {
        let sampler = DiscreteGaussian::new(BigURational::from_integer(BigUint::from(5u8)));

        // check samples are consistent
        let mut rng = SeedStreamSha3::from_seed(Seed::from_bytes([0u8; 16]));
        let samples: Vec<i8> = (0..10)
            .map(|_| i8::try_from(sampler.sample(&mut rng)).unwrap())
            .collect();
        let samples1: Vec<i8> = (0..10)
            .map(|_| i8::try_from(sampler.sample(&mut rng)).unwrap())
            .collect();
        assert_eq!(samples, vec!(3, 8, -7, 1, 2, 10, 8, -3, 0, 0));
        assert_eq!(samples1, vec!(-1, 2, 5, -1, -1, 3, 3, -1, -1, 3));

        // test zcdp constructor
        let zcdp = ZCdpDiscreteGaussian {
            budget: ZeroConcentratedDifferentialPrivacyBudget {
                epsilon: BigURational::new(1u8.into(), 5u8.into()),
            },
        };
        let sampler1 = zcdp.create_distribution(BigURational::from_integer(1u8.into()));
        let mut rng1 = SeedStreamSha3::from_seed(Seed::from_bytes([0u8; 16]));
        let samples2: Vec<i8> = (0..10)
            .map(|_| i8::try_from(sampler1.sample(&mut rng1)).unwrap())
            .collect();
        assert_eq!(samples2, samples);
    }
}
