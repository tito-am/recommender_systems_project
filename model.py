"""NeuMF model: GMF branch + MLP branch combined, trained directly end-to-end.

Same architecture as the GMF/MLP branches in the course notebook, but without
the separate pretraining + weight-transfer step -- everything trains together
with one Adam optimizer and binary cross-entropy loss.
"""
from tensorflow import keras
from tensorflow.keras import layers

EMBEDDING_DIM = 32
HIDDEN_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.2
L2_REG = 1e-6


def build_ncf_model(
    n_users: int,
    n_items: int,
    embedding_dim: int = EMBEDDING_DIM,
    hidden_units=HIDDEN_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    l2_reg: float = L2_REG,
    learning_rate: float = 1e-3,
) -> keras.Model:
    user_input = keras.Input(shape=(), name="user_idx", dtype="int32")
    item_input = keras.Input(shape=(), name="item_idx", dtype="int32")

    # GMF branch: element-wise product of user/item embeddings
    gmf_user = layers.Embedding(n_users, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l2_reg))(user_input)
    gmf_item = layers.Embedding(n_items, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l2_reg))(item_input)
    gmf_vector = layers.Multiply()([layers.Flatten()(gmf_user), layers.Flatten()(gmf_item)])

    # MLP branch: concatenated embeddings through a small feed-forward stack
    mlp_user = layers.Embedding(n_users, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l2_reg))(user_input)
    mlp_item = layers.Embedding(n_items, embedding_dim, embeddings_regularizer=keras.regularizers.l2(l2_reg))(item_input)
    x = layers.Concatenate()([layers.Flatten()(mlp_user), layers.Flatten()(mlp_item)])
    for units in hidden_units:
        x = layers.Dense(units, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = layers.Dropout(dropout_rate)(x)

    combined = layers.Concatenate()([gmf_vector, x])
    output = layers.Dense(1, activation="sigmoid", name="out")(combined)

    model = keras.Model(inputs={"user_idx": user_input, "item_idx": item_input}, outputs=output, name="ncf")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model
