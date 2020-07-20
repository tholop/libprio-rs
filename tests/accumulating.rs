use libprio_rs;
use libprio_rs::client::*;
use libprio_rs::finite_field::Field;
use libprio_rs::server::*;

#[test]
fn accumulation() {
    let dim = 123;
    let number_of_clients = 100;

    let mut reference_count = vec![0u32; dim];

    let mut server1 = Server::new(dim, true);
    let mut server2 = Server::new(dim, false);

    let mut client_mem = Client::new(dim).unwrap();

    use rand::Rng;
    let mut rng = rand::thread_rng();
    for _ in 0..number_of_clients {
        // some random data
        let data = (0..dim)
            .map(|_| Field::from(rng.gen_range(0, 2)))
            .collect::<Vec<Field>>();

        // update reference count
        for (r, d) in reference_count.iter_mut().zip(data.iter()) {
            *r += u32::from(*d);
        }

        let (share1, share2) = client_mem.encode_simple(&data);

        let eval_at = server1.choose_eval_at();

        let v1 = server1
            .generate_verification_message(eval_at, &share1)
            .unwrap();
        let v2 = server2
            .generate_verification_message(eval_at, &share2)
            .unwrap();

        assert_eq!(server1.aggregate(&share1, &v1, &v2), true);
        assert_eq!(server2.aggregate(&share2, &v1, &v2), true);
    }

    let total1 = server1.total_shares();
    let total2 = server2.total_shares();

    let reconstructed = libprio_rs::util::reconstruct_shares(total1, total2).unwrap();
    assert_eq!(reconstructed, reference_count);
}
