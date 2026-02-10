# RCMCL
This repository contains the code of RCMCL
```
computer configuration：
    CPU: i7-12700KF
    GPU: RTX 4060Ti
    internal storage: 32GB

The required Python libraries:
    NumPy
    PyTorch
    Scikit-learn
    SciPy

All comparison algorithms employ this conflict construction method：
"
for test_t in range(test_time):
        index = np.arange(num_samples)
        np.random.shuffle(index)
        train_index, test_index = index[:int(0.8 * num_samples)], index[int(0.8 * num_samples):]
        dataset.postprocessing(test_index, addNoise=True, sigma=0.5, ratio_noise=0.1, addConflict=True, ratio_conflict=0.4)
        ......
"

Clicking on "Run main.py" will enable the successful execution of the code.
```
