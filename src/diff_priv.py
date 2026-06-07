import numpy as np
from diffprivlib.mechanisms import GaussianAnalytic


EPSILON = 1.0
DELTA = 1e-5
SENSITIVITY = 1.0

data = np.load("../data/embeddings/pan15_eng_sentence-transformers__all-MiniLM-L6-v2_mean.npz", allow_pickle=True)

x_train = data["x_train"]
x_test = data["x_test"]


for eps in [1.0, 5.0, 10.0]:
    mech = GaussianAnalytic(
        epsilon=eps,
        delta=DELTA,
        sensitivity=SENSITIVITY
    )

    x_train_private = np.empty_like(x_train)

    for i in range(x_train.shape[0]):
        for j in range(x_train.shape[1]):
            x_train_private[i, j] = mech.randomise(float(x_train[i, j]))

    x_test_private = np.empty_like(x_test)

    for i in range(x_test.shape[0]):
        for j in range(x_test.shape[1]):
            x_test_private[i, j] = mech.randomise(float(x_test[i, j]))
    

    np.savez_compressed(
    f"../data/embeddings/embeddings_eps_{eps}.npz",
    x_train=x_train_private,
    x_test=x_test_private,
    train_author_ids=data["train_author_ids"],
    test_author_ids=data["test_author_ids"],
    y_train_gender=data["y_train_gender"],
    y_test_gender=data["y_test_gender"],
    y_train_age=data["y_train_age"],
    y_test_age=data["y_test_age"],
    )