# coding=utf-8
"""
Evaluation of caching performance using loading / saving dict by pickle module.

@author: sklykov

@license: The Unlicense

"""
import numpy as np
import pickle
from pathlib import Path
import time
from joblib import dump, load
import hashlib

# Run as the main script
if __name__ == '__main__':
    # Dev. Note: after introduction of hashing of keys, saving reduced to
    # By using common pickle it takes ~ 15 secs for 50_000 values for raw keys as arrays
    script_path = Path(__file__).resolve().parent; n_entries = 100_000
    cache_dict = {}; cache_dict["Entry 1"] = {}; cache_dict["Entry 1"]["A"] = {}; cache_dict["Entry 1"]["B"] = {}
    test_joblib_save = False  # flag to compare saving by pickle and joblib methods. Note: joblib much less effective than pickle
    # Generate huge dict mocking the predefined structure
    rng = np.random.default_rng()  # creates a random generator
    probe_key = tuple(np.round(rng.uniform(-2.0, 2.0, size=(91, )), 6))
    probe_value = rng.integers(0, 2**12-2, size=(64, ))
    keys_to_retrieve = []
    t1 = time.perf_counter()
    for i in range(n_entries):
        key = np.round(rng.uniform(-2.0, 2.0, size=(91, )), 6).astype(np.float64)
        key = key.astype('>f8', copy=False)  # prepare key with special type: big-endian, 8 bytes for float
        key_h = hashlib.blake2b(key.tobytes(), digest_size=8).hexdigest()  # get hash sum from an array
        value = rng.integers(0, 2**12-2, size=(64, ))
        cache_dict["Entry 1"]["A"][key_h] = value
        # prepare also data for checking retrieving performance
        if i % 5 == 0:  # save each 5th key for checking retrieving performance
            keys_to_retrieve.append(key_h)
        elif i % 8 == 0:  # generate another random key not presented in dict
            key = np.round(rng.uniform(-2.0, 2.0, size=(91, )), 6).astype(np.float64)
            key = key.astype('>f8', copy=False)
            keys_to_retrieve.append(hashlib.blake2b(key.tobytes(), digest_size=8).hexdigest())
    print(f"Generation of {n_entries} entries takes sec-s:", (round(time.perf_counter() - t1, 2)))

    # Save all values and measure performance
    t1 = time.perf_counter()
    if not test_joblib_save:
        storage_path = script_path.joinpath("cached_dict.pkl")
        with open(str(storage_path), "wb") as data_file:
            pickle.dump(cache_dict, data_file, protocol=5)
        print(f"Saving of {n_entries} entries by pickle takes sec-s:", (round(time.perf_counter() - t1, 2)))
    else:
        storage_path = script_path.joinpath("cached_dict.joblib")
        with open(str(storage_path), "wb") as data_file:
            dump(cache_dict, data_file, compress=0)
        print(f"Saving of {n_entries} entries by joblib takes sec-s:", (round(time.perf_counter() - t1, 2)))
    del cache_dict  # clean up

    if storage_path.exists():
        # Load all values and measure performance
        t1 = time.perf_counter()
        with open(str(storage_path), "rb") as data_file:
            if not test_joblib_save:
                load_dict = pickle.load(data_file)
            else:
                load_dict = load(data_file)
        print(f"Loading of {n_entries} entries takes sec-s:", (round(time.perf_counter() - t1, 2)))

        # Assess search performance
        timings = [0.0]*len(keys_to_retrieve)
        for i in range(len(keys_to_retrieve)):
            t1 = time.perf_counter_ns()
            if keys_to_retrieve[i] in load_dict:
                value = load_dict[keys_to_retrieve[i]]
            timings[i] = round(1E-6*(time.perf_counter_ns() - t1), 3)
        print("Average retrieve of value by key time in ms:", round(np.mean(timings), 3))

        # clean up saved huge pickle file
        storage_path.unlink()
