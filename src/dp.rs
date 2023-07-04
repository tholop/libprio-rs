// SPDX-License-Identifier: MPL-2.0

//! Differential privacy (DP) primitives.
use std::fmt::Debug;

#[cfg(feature = "experimental")]
use crate::dp::samplers::sample_discrete_gaussian;
#[cfg(feature = "experimental")]
use num_bigint::{BigInt, BigUint};
#[cfg(feature = "experimental")]
use num_rational::Ratio;
#[cfg(feature = "experimental")]
use rand::{distributions::Distribution, Rng};

#[cfg(feature = "experimental")]
mod samplers;

#[cfg(feature = "experimental")]
/// Alias for arbitrary precision unsigned rationals.
pub type BigURational = Ratio<BigUint>;

/// Marker trait for differential privacy budgets (regardless of the specific accounting method).
pub trait DifferentialPrivacyBudget {}

/// Marker trait for differential privacy scalar noise distributions.
pub trait DifferentialPrivacyDistribution {}

/// Zero-concentrated differential privacy (zCDP) budget as defined in [[BS16]].
///
/// [BS16]: https://arxiv.org/pdf/1605.02065.pdf
pub struct ZeroConcentratedDifferentialPrivacyBudget {
    /// Parameter `epsilon`, using the notation from [[CKS20]] where `rho = (epsilon**2)/2`
    /// for a `rho`-zCDP budget.
    ///
    /// [CKS20]: https://arxiv.org/pdf/2004.00010.pdf
    pub epsilon: BigURational,
}

impl DifferentialPrivacyBudget for ZeroConcentratedDifferentialPrivacyBudget {}

/// Samples `BigInt` numbers according to the discrete Gaussian distribution with mean zero.
/// The distribution is defined over the integers, represented by arbitrary-precision integers.
/// The sampling procedute follows [CKS20].
///
/// [CKS20](https://arxiv.org/abs/2004.00010)
#[derive(Clone, Debug)]
pub struct DiscreteGaussian {
    /// The standard deviation of the distribution.
    std: BigURational,
}

impl DiscreteGaussian {
    /// Create a new sampler from the Discrete Gaussian Distribution with the given
    /// standard deviation and mean zero.
    pub fn new(std: BigURational) -> DiscreteGaussian {
        DiscreteGaussian { std }
    }
}

impl Distribution<BigInt> for DiscreteGaussian {
    fn sample<R>(&self, rng: &mut R) -> BigInt
    where
        R: Rng + ?Sized,
    {
        sample_discrete_gaussian(&self.std, rng) // TODO if we end up using BigURational the sampler should use it
    }
}

impl DifferentialPrivacyDistribution for DiscreteGaussian {}

/// A zCDP budget used to create a Discrete Gaussian distribution
pub struct ZCdpDiscreteGaussian {
    budget: ZeroConcentratedDifferentialPrivacyBudget,
}

/// Strategy to make aggregate shares differentially private, e.g. by adding noise from a specific
/// type of distribution instantiated with a given DP budget
pub trait DifferentialPrivacyStrategy {
    type Budget: DifferentialPrivacyBudget;
    type Distribution: DifferentialPrivacyDistribution;
    type Sensitivity;

    fn from_budget(b: Self::Budget) -> Self;
    fn create_distribution(&self, s: Self::Sensitivity) -> Self::Distribution;
}

impl DifferentialPrivacyStrategy for ZCdpDiscreteGaussian {
    type Budget = ZeroConcentratedDifferentialPrivacyBudget;
    type Distribution = DiscreteGaussian;
    type Sensitivity = BigURational;

    fn from_budget(b: Self::Budget) -> ZCdpDiscreteGaussian {
        ZCdpDiscreteGaussian { budget: b }
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
        let sampler = DiscreteGaussian {
            std: BigURational::from_integer(BigUint::from(5u8)),
        };

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
