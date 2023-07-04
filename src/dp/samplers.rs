//! Implementation of a sampler from the Discrete Gaussian Distribution.
//!
//! Follows
//!     Clément Canonne, Gautam Kamath, Thomas Steinke. The Discrete Gaussian for Differential Privacy. 2020.
//!     <https://arxiv.org/abs/2004.00010>

// Copyright (c) 2022 President and Fellows of Harvard College
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// This file incorporates work covered by the following copyright and
// permission notice:
//
//   Copyright 2020 Thomas Steinke
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

//   The following code is adapted from the opendp implementation to reduce dependencies:
//       https://github.com/opendp/opendp/blob/main/rust/src/traits/samplers/cks20

use num_bigint::{BigInt, BigUint};
use num_traits::{One, Zero};
use rand::{distributions::Distribution, distributions::Uniform, Rng};

use super::BigURational;

/// Sample from the Bernoulli(1/2) distribution.
///
/// `sample_bernoulli_standard(rng)` returns numbers distributed as $Bernoulli(1/2)$.
/// using the given random number generator for base randomness.
fn sample_bernoulli_standard<R: Rng + ?Sized>(rng: &mut R) -> bool {
    let mut buffer = [0u8; 1];
    rng.fill_bytes(&mut buffer);
    buffer[0] & 1 == 1
}

/// Sample from the Bernoulli(gamma) distribution, where $gamma /leq 1$.
///
/// `sample_bernoulli_frac(gamma, rng)` returns numbers distributed as $Bernoulli(gamma)$.
/// using the given random number generator for base randomness.
fn sample_bernoulli_frac<R: Rng + ?Sized>(gamma: &BigURational, rng: &mut R) -> bool {
    let d = gamma.denom();
    assert!(!d.is_zero());
    assert!(gamma <= &BigURational::one());

    // sample uniform biguint in [0,d)
    let s = rng.gen_range(BigUint::zero()..d.clone());

    s < *gamma.numer()
}

/// Sample from the Bernoulli(exp(-gamma)) distribution, where $gamma \leq 1$.
///
/// `sample_bernoulli_exp1(gamma, rng)` returns numbers distributed as $Bernoulli(exp(-gamma))$.
/// using the given random number generator for base randomness.
fn sample_bernoulli_exp1<R: Rng + ?Sized>(gamma: &BigURational, rng: &mut R) -> bool {
    assert!(!gamma.denom().is_zero());
    assert!(gamma <= &BigURational::one());
    let mut k = BigUint::one();
    loop {
        if sample_bernoulli_frac(&(gamma / k.clone()), rng) {
            k += 1u8;
        } else {
            return !(k % BigUint::from(2u8)).is_zero();
        }
    }
}

/// Sample from the Bernoulli(exp(-gamma)) distribution.
///
/// `sample_bernoulli_exp(gamma, rng)` returns numbers distributed as $Bernoulli(exp(-gamma))$,
/// using the given random number generator for base randomness.
fn sample_bernoulli_exp<R: Rng + ?Sized>(gamma: &BigURational, rng: &mut R) -> bool {
    assert!(!gamma.denom().is_zero());
    // Sample floor(n/d) independent Bernoulli(exp(-1))
    // If all are 1, return Bernoulli(exp(-(gamma-floor(gamma))))
    let mut gamma: BigURational = gamma.clone();
    while BigURational::one() < gamma {
        if !sample_bernoulli_exp1(&BigURational::one(), rng) {
            return false;
        }
        gamma -= BigURational::one();
    }
    sample_bernoulli_exp1(&gamma, rng)
}

/// Sample from the geometric distribution with parameter 1 - exp(-gamma) (slow).
///
/// `sample_geometric_exp_slow(gamma, rng)` returns numbers distributed according to
/// $Geometric(1 - exp(-gamma))$, using the given random number generator for base randomness.
fn sample_geometric_exp_slow<R: Rng + ?Sized>(gamma: &BigURational, rng: &mut R) -> BigUint {
    assert!(!gamma.denom().is_zero());
    let mut k = BigUint::zero();
    loop {
        if sample_bernoulli_exp(gamma, rng) {
            k += 1u8;
        } else {
            return k;
        }
    }
}

/// Sample from the geometric distribution  with parameter 1 - exp(-gamma) (fast).
///
/// `sample_geometric_exp_fast(gamma, rng)` returns numbers distributed according to
/// $Geometric(1 - exp(-gamma))$, using the given random number generator for base randomness.
fn sample_geometric_exp_fast<R: Rng + ?Sized>(gamma: &BigURational, rng: &mut R) -> BigUint {
    let d = gamma.denom();
    assert!(!d.is_zero());
    if gamma.is_zero() {
        return BigUint::zero();
    }

    // sample uniform biguint in [0,d)

    let usampler = Uniform::new(BigUint::zero(), d);
    let mut u = usampler.sample(rng);

    while !sample_bernoulli_exp(&BigURational::new(u.clone(), d.clone()), rng) {
        u = usampler.sample(rng);
    }

    let v2 = sample_geometric_exp_slow(&BigURational::one(), rng);
    v2 * d + u / gamma.numer()
}

/// Sample from the discrete laplace distribution.
///
/// `sample_discrete_laplace(scale, rng)` returns numbers distributed according to
/// $\mathcal{L}_\mathbb{Z}(0, scale)$, using the given random number generator for base randomness.
///
/// # Citation
/// * [CKS20 The Discrete Gaussian for Differential Privacy](https://arxiv.org/abs/2004.00010)
pub fn sample_discrete_laplace<R: Rng + ?Sized>(scale: &BigURational, rng: &mut R) -> BigInt {
    let (n, d) = (scale.numer(), scale.denom());
    assert!(!d.is_zero());
    if n.is_zero() {
        return BigInt::zero();
    }

    loop {
        let negative = sample_bernoulli_standard(rng);
        let magnitude: BigInt = sample_geometric_exp_fast(&scale.recip(), rng).into();
        if negative || !magnitude.is_zero() {
            return if negative { -magnitude } else { magnitude };
        }
    }
}

/// Sample from the discrete gaussian distribution.
///
/// `sample_discrete_gaussian(sigma, rng)` returns `BigInt` numbers distributed as
/// $\mathcal{N}_\mathbb{Z}(0, sigma^2)$,
/// using the given random number generator for base randomness.
///
/// # Citation
/// * [CKS20 The Discrete Gaussian for Differential Privacy](https://arxiv.org/abs/2004.00010)
pub fn sample_discrete_gaussian<R: Rng + ?Sized>(sigma: &BigURational, rng: &mut R) -> BigInt {
    assert!(!sigma.denom().is_zero());
    if sigma.is_zero() {
        return 0.into();
    }
    let t = sigma.floor() + BigUint::one();
    loop {
        let y = sample_discrete_laplace(&t, rng);

        // absolute value without type conversion
        let y_abs: BigURational = BigUint::new(y.to_u32_digits().1).into();

        let sub = sigma.pow(2) / t.clone();
        let fact = (sigma.pow(2) * BigUint::from(2u8)).recip();

        let prob: BigURational = if y_abs < sub {
            (sub - y_abs).pow(2) * fact
        } else {
            (y_abs - sub).pow(2) * fact
        };
        if sample_bernoulli_exp(&prob, rng) {
            return y;
        }
    }
}

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
