use crate::finite_field::Field;

pub fn vector_with_length(len: usize) -> Vec<Field> {
    vec![Field::from(0); len]
}

pub fn proof_length(dimension: usize) -> usize {
    // number of data items + number of zero terms + N
    dimension + 3 + (dimension + 1).next_power_of_two()
}

pub struct UnpackedProof<'a> {
    pub data: &'a [Field],
    pub f0: &'a Field,
    pub g0: &'a Field,
    pub h0: &'a Field,
    pub points_h_packed: &'a [Field],
}

pub struct UnpackedProofMut<'a> {
    pub data: &'a mut [Field],
    pub f0: &'a mut Field,
    pub g0: &'a mut Field,
    pub h0: &'a mut Field,
    pub points_h_packed: &'a mut [Field],
}

pub fn unpack_proof(proof: &[Field], dimension: usize) -> Option<UnpackedProof> {
    // check the proof length
    if proof.len() != proof_length(dimension) {
        return None;
    }
    // split share into components
    let (data, rest) = proof.split_at(dimension);
    let (zero_terms, points_h_packed) = rest.split_at(3);
    if let [f0, g0, h0] = zero_terms {
        let unpacked = UnpackedProof {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        };
        Some(unpacked)
    } else {
        None
    }
}

pub fn unpack_proof_mut(proof: &mut [Field], dimension: usize) -> Option<UnpackedProofMut> {
    // check the share length
    if proof.len() != proof_length(dimension) {
        return None;
    }
    // split share into components
    let (data, rest) = proof.split_at_mut(dimension);
    let (zero_terms, points_h_packed) = rest.split_at_mut(3);
    if let [f0, g0, h0] = zero_terms {
        let unpacked = UnpackedProofMut {
            data,
            f0,
            g0,
            h0,
            points_h_packed,
        };
        Some(unpacked)
    } else {
        None
    }
}

pub fn serialize(data: &[Field]) -> Vec<u8> {
    let field_size = std::mem::size_of::<Field>();
    let mut vec = Vec::with_capacity(data.len() * field_size);

    for elem in data.iter() {
        let int = u32::from(*elem);
        vec.extend(int.to_le_bytes().iter());
    }

    vec
}

pub fn deserialize(data: &[u8]) -> Vec<Field> {
    let field_size = std::mem::size_of::<Field>();

    let mut vec = Vec::with_capacity(data.len() / field_size);
    use std::convert::TryInto;

    for chunk in data.chunks_exact(field_size) {
        let integer = u32::from_le_bytes(chunk.try_into().unwrap());
        vec.push(Field::from(integer));
    }

    vec
}

pub fn reconstruct_shares(share1: &[Field], share2: &[Field]) -> Option<Vec<Field>> {
    if share1.len() != share2.len() {
        return None;
    }

    let mut reconstructed = vector_with_length(share1.len());

    for (r, (s1, s2)) in reconstructed
        .iter_mut()
        .zip(share1.iter().zip(share2.iter()))
    {
        *r = *s1 + *s2;
    }

    Some(reconstructed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secret_share(share: &mut [Field]) -> Vec<Field> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut random = vec![0u32; share.len()];
        let mut share2 = vector_with_length(share.len());

        rng.fill(&mut random[..]);

        for (r, f) in random.iter().zip(share2.iter_mut()) {
            *f = Field::from(*r);
        }

        for (f1, f2) in share.iter_mut().zip(share2.iter()) {
            *f1 -= *f2;
        }

        share2
    }

    #[test]
    fn test_unpack_share() {
        let dim = 15;
        let len = proof_length(dim);

        let mut share = vec![Field::from(0); len];
        let unpacked = unpack_proof_mut(&mut share, dim).unwrap();
        *unpacked.f0 = Field::from(12);
        assert_eq!(share[dim], 12);
    }

    #[test]
    fn secret_sharing() {
        let mut share1 = vector_with_length(10);
        share1[3] = 21.into();
        share1[8] = 123.into();

        let original_data = share1.clone();

        let share2 = secret_share(&mut share1);

        let reconstructed = reconstruct_shares(&share1, &share2).unwrap();
        assert_eq!(reconstructed, original_data);
    }

    #[test]
    fn serialization() {
        let field = [Field::from(1), Field::from(0x99997)];
        let bytes = serialize(&field);
        let field_deserialized = deserialize(&bytes);
        assert_eq!(field_deserialized, field);
    }
}
