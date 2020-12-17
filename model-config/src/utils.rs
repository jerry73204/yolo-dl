use crate::common::*;

pub fn hash_vec_indexset<T, H>(set: &IndexSet<T>, state: &mut H)
where
    T: Hash,
    H: Hasher,
{
    let set: Vec<_> = set.iter().collect();
    set.hash(state);
}

pub fn hash_vec_indexmap<K, V, H>(set: &IndexMap<K, V>, state: &mut H)
where
    K: Hash,
    V: Hash,
    H: Hasher,
{
    let map: Vec<_> = set.iter().collect();
    map.hash(state);
}

pub fn empty_vec<T>() -> Vec<T> {
    vec![]
}
